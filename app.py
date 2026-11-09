import os, json, math
from datetime import date, datetime
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

st.set_page_config(page_title="Football AI Predictor", page_icon="⚽", layout="wide")

LEAGUES = {
    "Premier League": "E0", "Liga": "SP1", "Bundesliga": "D1",
    "Serie A": "I1", "Ligue 1": "F1", "Championship": "E1",
}
SEASONS = ["2627", "2526", "2425"]

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_league_data(code: str):
    completed, fixtures, errors = [], [], []
    for season in SEASONS:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 FootballAIPredictor/1.0"})
            r.raise_for_status()
            df = pd.read_csv(BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
            required = ["Date", "HomeTeam", "AwayTeam"]
            if not all(c in df.columns for c in required):
                errors.append(f"{season}: format CSV inattendu")
                continue
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df = df.dropna(subset=required).copy()
            df["FTHG"] = pd.to_numeric(df.get("FTHG"), errors="coerce")
            df["FTAG"] = pd.to_numeric(df.get("FTAG"), errors="coerce")
            played = df.dropna(subset=["FTHG", "FTAG"]).copy()
            if len(played):
                played["FTHG"] = played["FTHG"].astype(int)
                played["FTAG"] = played["FTAG"].astype(int)
                completed.append(played)
            future = df[df["FTHG"].isna() | df["FTAG"].isna()].copy()
            if len(future): fixtures.append(future)
        except Exception as e:
            errors.append(f"{season}: {type(e).__name__}")
    if not completed:
        raise RuntimeError("Aucun résultat récent n'a pu être téléchargé. Vérifie ta connexion ou réessaie plus tard.")
    matches = pd.concat(completed, ignore_index=True).drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    matches = matches.sort_values("Date").reset_index(drop=True)
    upcoming = pd.concat(fixtures, ignore_index=True) if fixtures else pd.DataFrame()
    if len(upcoming):
        upcoming = upcoming[upcoming["Date"].dt.date >= date.today()].sort_values("Date")
        upcoming = upcoming.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return matches, upcoming, errors

class DixonColes:
    def __init__(self, xi=0.0018): self.xi = xi
    def fit(self, df):
        self.teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        self.idx = {t:i for i,t in enumerate(self.teams)}; n=len(self.teams)
        h=df.HomeTeam.map(self.idx).to_numpy(); a=df.AwayTeam.map(self.idx).to_numpy()
        x=df.FTHG.to_numpy(dtype=int); y=df.FTAG.to_numpy(dtype=int)
        days=(df.Date.max()+pd.Timedelta(days=1)-df.Date).dt.days.to_numpy().clip(min=0)
        w=np.exp(-self.xi*days); const=gammaln(x+1)+gammaln(y+1)
        def unpack(p): return p[:n]-p[:n].mean(),p[n:2*n],p[2*n],p[2*n+1]
        def nll(p):
            att,deff,home,rho=unpack(p); lh=home+att[h]-deff[a]; la=att[a]-deff[h]
            lam=np.exp(lh); mu=np.exp(la); tau=np.ones(len(x))
            z=(x==0)&(y==0); tau[z]=1-lam[z]*mu[z]*rho
            z=(x==0)&(y==1); tau[z]=1+lam[z]*rho
            z=(x==1)&(y==0); tau[z]=1+mu[z]*rho
            z=(x==1)&(y==1); tau[z]=1-rho
            return -(w*(np.log(np.clip(tau,1e-10,None))+x*lh-lam+y*la-mu-const)).sum()
        p0=np.r_[np.zeros(n),np.full(n,-.2),.25,-.05]
        res=minimize(nll,p0,method="L-BFGS-B",bounds=[(-3,3)]*(2*n)+[(-1,1),(-.3,.3)])
        self.att,self.deff,self.home,self.rho=unpack(res.x); self.converged=res.success
        return self
    def predict(self, home, away):
        if home not in self.idx or away not in self.idx: raise ValueError("Une équipe est absente des résultats chargés.")
        i,j=self.idx[home],self.idx[away]
        lam=float(np.exp(self.home+self.att[i]-self.deff[j])); mu=float(np.exp(self.att[j]-self.deff[i]))
        g=np.arange(11); m=np.outer(poisson.pmf(g,lam),poisson.pmf(g,mu))
        m[0,0]*=1-lam*mu*self.rho; m[0,1]*=1+lam*self.rho; m[1,0]*=1+mu*self.rho; m[1,1]*=1-self.rho
        m/=m.sum(); total=g[:,None]+g[None,:]
        scores=sorted([(int(x),int(y),float(m[x,y])) for x in g for y in g],key=lambda q:q[2],reverse=True)[:5]
        return {"xg_home":lam,"xg_away":mu,"p_home":float(np.tril(m,-1).sum()),"p_draw":float(np.trace(m)),"p_away":float(np.triu(m,1).sum()),"over25":float(m[total>=3].sum()),"under25":float(m[total<3].sum()),"btts":float(m[1:,1:].sum()),"scores":scores}

def openrouter(api_key, model, system, prompt):
    url="https://openrouter.ai/api/v1/chat/completions"
    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json","X-Title":"Football AI Predictor"}
    payload={"model":model,"temperature":0.2,"messages":[{"role":"system","content":system},{"role":"user","content":prompt}]}
    r=requests.post(url,headers=headers,json=payload,timeout=90); r.raise_for_status()
    data=r.json()
    return data["choices"][0]["message"]["content"]

def search_news(key, query):
    if not key: return []
    r=requests.post("https://api.tavily.com/search",json={"api_key":key,"query":query,"search_depth":"advanced","max_results":5},timeout=40)
    r.raise_for_status(); return r.json().get("results",[])

def pct(x): return f"{x*100:.1f}%"

st.markdown("""<style>
.stApp{background:#0b1220;color:#eef4ff}.block-container{padding-top:2rem;max-width:1200px}
[data-testid="stMetric"]{background:#142238;padding:14px;border-radius:12px;border:1px solid #253955}
div.stButton>button[kind="primary"]{background:#16a36a;border:0;color:white;font-weight:700}
</style>""",unsafe_allow_html=True)
st.title("⚽ FOOTBALL AI PREDICTOR")
st.caption("Dixon–Coles • 3 analystes IA • recherche d'actualité • arbitre final")
st.info("Les résultats des championnats sont téléchargés automatiquement. Pour l'analyse multi-IA, une clé OpenRouter est nécessaire. Sans clé, le moteur statistique reste utilisable.")

with st.sidebar:
    st.header("⚙️ Paramètres")
    league=st.selectbox("Championnat",list(LEAGUES.keys()))
    st.caption("Données : saisons 2024/25 à 2026/27, selon disponibilité.")
    st.divider(); st.subheader("Agents IA (facultatif)")
    api_key=st.text_input("Clé OpenRouter",type="password",help="Une seule clé pour les différents agents. Les appels peuvent être payants.")
    m_stats=st.text_input("Agent statistique","openai/gpt-4o-mini")
    m_tactics=st.text_input("Agent tactique","google/gemini-2.5-flash")
    m_news=st.text_input("Agent actualités","openai/gpt-4o-mini")
    m_judge=st.text_input("Superviseur","anthropic/claude-3.5-sonnet")
    tavily=st.text_input("Clé Tavily (recherche Web)",type="password",help="Facultative : permet la recherche d'actualités et de sources récentes.")
    st.caption("Ne partage jamais tes clés API. Tu peux utiliser uniquement le moteur statistique sans clé.")

if st.button("🔄 Actualiser les données",use_container_width=True):
    fetch_league_data.clear(); st.rerun()

try:
    with st.spinner(f"Téléchargement des résultats de {league}…"):
        data, fixtures, errs=fetch_league_data(LEAGUES[league])
    model=DixonColes().fit(data)
    st.success(f"{len(data)} matchs chargés • {len(model.teams)} équipes • dernière rencontre : {data.Date.max():%d/%m/%Y}")
    if errs: st.caption("Certaines saisons indisponibles : "+" ; ".join(errs))
except Exception as e:
    st.error(str(e)); st.stop()

st.subheader("1. Choisir un match")
upcoming_options=[]
if len(fixtures):
    for _,r in fixtures.iterrows():
        if r.HomeTeam in model.idx and r.AwayTeam in model.idx:
            upcoming_options.append((f"{r.Date:%d/%m/%Y} — {r.HomeTeam} vs {r.AwayTeam}",r.HomeTeam,r.AwayTeam,r.Date.date()))
if upcoming_options:
    labels=[x[0] for x in upcoming_options]
    chosen=st.selectbox("Matchs à venir détectés dans le calendrier",labels)
    item=upcoming_options[labels.index(chosen)]; default_home,default_away,default_date=item[1],item[2],item[3]
else:
    st.caption("Aucun calendrier futur détecté dans le fichier du championnat : sélectionne les équipes manuellement.")
    default_home=model.teams[0]; default_away=model.teams[1] if len(model.teams)>1 else model.teams[0]; default_date=date.today()
c1,c2,c3=st.columns(3)
with c1: home=st.selectbox("Domicile",model.teams,index=model.teams.index(default_home))
with c2: away=st.selectbox("Extérieur",model.teams,index=model.teams.index(default_away))
with c3: match_date=st.date_input("Date",value=default_date)
context=st.text_area("Informations complémentaires (facultatif)",placeholder="Absences confirmées, enjeu, match aller/retour…")
with st.expander("Cotes bookmaker (facultatif)"):
    o1,oX,o2=st.columns(3)
    with o1: odd1=st.number_input("Cote 1",min_value=1.01,value=2.0,step=.01)
    with oX: oddX=st.number_input("Cote X",min_value=1.01,value=3.4,step=.01)
    with o2: odd2=st.number_input("Cote 2",min_value=1.01,value=3.6,step=.01)

if st.button("🚀 ANALYSER LE MATCH",type="primary",use_container_width=True):
    if home==away: st.error("Choisis deux équipes différentes.")
    else:
        dc=model.predict(home,away)
        st.session_state["dc_result"]={"dc":dc,"home":home,"away":away,"date":str(match_date),"context":context,"league":league}
        if api_key:
            matchdata={"championnat":league,"match":f"{home} (domicile) vs {away} (extérieur)","date":str(match_date),"contexte":context,"dixon_coles":{k:v for k,v in dc.items() if k!="scores"},"scores_probables":[{"score":f"{x}-{y}","probabilite":pct(p)} for x,y,p in dc["scores"]],"cotes_saisies":{"1":odd1,"X":oddX,"2":odd2}}
            results=[]; prog=st.progress(0)
            roles=[("Analyste statistique",m_stats,"Analyse les probabilités, la forme statistique disponible et les marchés. Distingue les faits des hypothèses."),("Analyste tactique",m_tactics,"Analyse les styles, les oppositions tactiques, le terrain et les facteurs sportifs. N'invente aucune absence ni composition."),("Veille actualités",m_news,"Vérifie les actualités et informations récentes. Si tu n'as pas d'accès de recherche, indique clairement les limites et n'invente pas de sources.")]
            web=[]
            if tavily:
                try: web=search_news(tavily,f"{home} vs {away} {match_date} injury team news suspensions recent form football")
                except Exception as e: st.warning(f"Recherche Web non disponible : {e}")
            matchdata["resultats_recherche"]=web
            for i,(name,mod,system) in enumerate(roles):
                with st.spinner(name+" en cours…"):
                    try: answer=openrouter(api_key,mod,system,"Analyse ce match avec prudence. Données disponibles :\n"+json.dumps(matchdata,ensure_ascii=False,default=str))
                    except Exception as e: answer=f"Agent indisponible : {e}"
                results.append({"agent":name,"model":mod,"answer":answer}); prog.progress((i+1)/4)
            with st.spinner("Le superviseur confronte les analyses…"):
                sys_prompt="Tu es le superviseur. Confronte les analyses, relève les contradictions et les données manquantes. Donne des probabilités finales cohérentes, les marchés les plus prudents, les arguments pour/contre, les risques et un niveau de confiance. Compare aux cotes si disponibles en tenant compte de la marge. N'invente aucune information et recommande l'abstention si nécessaire. Aucun pari n'est sûr."
                try: final=openrouter(api_key,m_judge,sys_prompt,json.dumps({"match":matchdata,"analyses":results},ensure_ascii=False,default=str))
                except Exception as e: final=f"Superviseur indisponible : {e}"
            st.session_state["ai_result"]={"agents":results,"final":final,"web":web}
        else:
            st.session_state.pop("ai_result",None)

if "dc_result" in st.session_state:
    r=st.session_state["dc_result"]; dc=r["dc"]
    st.divider(); st.header(f"2. Résultats — {r['home']} vs {r['away']}")
    st.caption(f"{r['league']} • {r['date']} • modèle Dixon–Coles entraîné sur {len(data)} matchs")
    a,b,c=st.columns(3); a.metric("Victoire domicile",pct(dc["p_home"])); b.metric("Match nul",pct(dc["p_draw"])); c.metric("Victoire extérieur",pct(dc["p_away"]))
    a,b,c=st.columns(3); a.metric("Plus de 2,5 buts",pct(dc["over25"])); b.metric("Moins de 2,5 buts",pct(dc["under25"])); c.metric("Les deux marquent",pct(dc["btts"]))
    st.write(f"**Buts attendus :** {dc['xg_home']:.2f} – {dc['xg_away']:.2f}")
    st.write("**Scores exacts les plus probables :** "+" • ".join(f"{x}-{y} ({pct(p)})" for x,y,p in dc["scores"]))
    if "ai_result" in st.session_state:
        ai=st.session_state["ai_result"]; st.header("3. Verdict multi-agents")
        st.markdown(ai["final"])
        for agent in ai["agents"]:
            with st.expander(agent["agent"]+" — "+agent["model"]): st.markdown(agent["answer"])
        if ai["web"]:
            st.subheader("Sources de recherche")
            for s in ai["web"]: st.markdown(f"- [{s.get('title','Source')}]({s.get('url','#')}) — {s.get('content','')[:350]}")
    else:
        st.info("Tu vois ici le pronostic statistique. Ajoute une clé OpenRouter dans les paramètres pour activer les agents IA et le superviseur.")
    report={"match":r,"dixon_coles":dc,"ai":st.session_state.get("ai_result")}
    st.download_button("⬇️ Télécharger le rapport",json.dumps(report,ensure_ascii=False,indent=2,default=str),file_name="rapport_football_ai.json",mime="application/json")

st.divider()
st.caption("Outil d'aide à la décision, jamais une garantie de gain. Les données peuvent être incomplètes ou retardées. Vérifie les compositions, les cotes et les informations avant toute décision. Ne mise jamais de somme que tu ne peux pas perdre.")
