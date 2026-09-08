from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

class ResearchStore:
    def __init__(self, path="srm_agent_runs/research.sqlite3"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.executescript("create table if not exists research_runs(id text primary key,payload text,stopped integer default 0,created_at text); create table if not exists memory(id text primary key,title text,content text,metadata text,created_at text);")
    def save(self,payload):
        rid=payload.get("run_id") or "research_"+uuid.uuid4().hex[:12]; payload["run_id"]=rid
        with sqlite3.connect(self.path) as db: db.execute("insert or replace into research_runs values(?,?,0,?)",(rid,json.dumps(payload),datetime.now(timezone.utc).isoformat()))
        return payload
    def get(self,rid):
        with sqlite3.connect(self.path) as db: row=db.execute("select payload,stopped from research_runs where id=?",(rid,)).fetchone()
        if not row:return None
        data=json.loads(row[0]); data["stopped"]=bool(row[1]); return data
    def stop(self,rid):
        with sqlite3.connect(self.path) as db: db.execute("update research_runs set stopped=1 where id=?",(rid,))
        return self.get(rid)
    def delete(self,rid):
        with sqlite3.connect(self.path) as db: db.execute("delete from research_runs where id=?",(rid,))
        return {"deleted":rid}
    def add_memory(self,title,content,metadata=None):
        mid="mem_"+uuid.uuid4().hex[:12]
        with sqlite3.connect(self.path) as db: db.execute("insert into memory values(?,?,?,?,?)",(mid,title,content,json.dumps(metadata or {}),datetime.now(timezone.utc).isoformat()))
        return {"id":mid,"title":title}
    def search_memory(self,q):
        terms=[x.lower() for x in q.split() if x]
        with sqlite3.connect(self.path) as db: rows=db.execute("select id,title,content,metadata,created_at from memory").fetchall()
        out=[]
        for r in rows:
            score=sum(t in (r[1]+" "+r[2]).lower() for t in terms)
            if score: out.append({"id":r[0],"title":r[1],"content":r[2],"metadata":json.loads(r[3]),"created_at":r[4],"score":score})
        return sorted(out,key=lambda x:x["score"],reverse=True)
    def get_memory(self,mid):
        with sqlite3.connect(self.path) as db: r=db.execute("select id,title,content,metadata,created_at from memory where id=?",(mid,)).fetchone()
        return {"id":r[0],"title":r[1],"content":r[2],"metadata":json.loads(r[3]),"created_at":r[4]} if r else None
    def delete_memory(self,mid):
        with sqlite3.connect(self.path) as db: db.execute("delete from memory where id=?",(mid,))
        return {"deleted":mid}
