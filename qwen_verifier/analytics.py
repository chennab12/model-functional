"""Minimal privacy-first, first-party Streamlit analytics backed by SQLite."""
from __future__ import annotations
import os,sqlite3,time
from pathlib import Path
import pandas as pd

def _path(path:str|None=None)->Path:
    return Path(path or os.getenv("ANALYTICS_DB_PATH","/tmp/model_verifier_analytics.db"))

def _connect(path:str|None=None):
    db=sqlite3.connect(_path(path),timeout=5)
    db.execute("CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY, started REAL, last_seen REAL, page_views INTEGER)")
    db.execute("CREATE TABLE IF NOT EXISTS events(ts REAL, session_id TEXT, section TEXT, event TEXT)")
    return db

def record_visit(session_id:str,path:str|None=None,now:float|None=None)->None:
    now=now or time.time()
    with _connect(path) as db:
        row=db.execute("SELECT session_id FROM sessions WHERE session_id=?",(session_id,)).fetchone()
        if row: db.execute("UPDATE sessions SET last_seen=?,page_views=page_views+1 WHERE session_id=?",(now,session_id))
        else: db.execute("INSERT INTO sessions VALUES(?,?,?,?)",(session_id,now,now,1))
        db.execute("INSERT INTO events VALUES(?,?,?,?)",(now,session_id,"site","page_view"))

def record_event(session_id:str,section:str,event:str,path:str|None=None,now:float|None=None)->None:
    with _connect(path) as db:db.execute("INSERT INTO events VALUES(?,?,?,?)",(now or time.time(),session_id,section[:80],event[:80]))

def analytics_summary(path:str|None=None)->dict:
    with _connect(path) as db:
        sessions=pd.read_sql_query("SELECT * FROM sessions",db)
        events=pd.read_sql_query("SELECT * FROM events",db)
    if sessions.empty:
        return {"sessions":0,"page_views":0,"avg_duration_seconds":0,"events":0,"daily":pd.DataFrame(),"heatmap":pd.DataFrame()}
    sessions["duration"]=sessions.last_seen-sessions.started
    if events.empty:heat=pd.DataFrame()
    else:
        dt=pd.to_datetime(events.ts,unit="s",utc=True); events["weekday"]=dt.dt.day_name().str[:3];events["hour"]=dt.dt.hour
        heat=events.pivot_table(index="weekday",columns="hour",values="event",aggfunc="count",fill_value=0).reindex(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]).fillna(0)
    daily=sessions.assign(day=pd.to_datetime(sessions.started,unit="s",utc=True).dt.date).groupby("day",as_index=False).agg(sessions=("session_id","count"),page_views=("page_views","sum"))
    return {"sessions":len(sessions),"page_views":int(sessions.page_views.sum()),"avg_duration_seconds":round(float(sessions.duration.mean()),1),"events":len(events),"daily":daily,"heatmap":heat}
