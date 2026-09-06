from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

class Store:
    def __init__(self, path="srm_agent_runs/agent.sqlite3"):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._connect() as db:
            db.executescript("""
            create table if not exists accounts(id text primary key, cash real not null, initial_cash real not null, created_at text not null);
            create table if not exists orders(id text primary key, account_id text not null, ticker text not null, side text not null, quantity real not null, price real not null, fee real not null, created_at text not null);
            create table if not exists runs(id text primary key, payload text not null, created_at text not null);
            """)
    def _connect(self): return sqlite3.connect(self.path)
    def save_run(self,payload):
        rid="run_"+uuid.uuid4().hex[:12]; now=datetime.now(timezone.utc).isoformat()
        with self._connect() as db: db.execute("insert into runs values(?,?,?)",(rid,json.dumps(payload),now))
        return rid
    def run(self,rid):
        with self._connect() as db: row=db.execute("select payload from runs where id=?",(rid,)).fetchone()
        return json.loads(row[0]) if row else None
    def create_account(self,cash):
        aid="paper_"+uuid.uuid4().hex[:12]; now=datetime.now(timezone.utc).isoformat()
        with self._connect() as db: db.execute("insert into accounts values(?,?,?,?)",(aid,float(cash),float(cash),now))
        return self.account(aid)
    def account(self,aid):
        with self._connect() as db:
            row=db.execute("select id,cash,initial_cash,created_at from accounts where id=?",(aid,)).fetchone()
            orders=db.execute("select id,ticker,side,quantity,price,fee,created_at from orders where account_id=? order by created_at",(aid,)).fetchall()
        if not row:return None
        positions={}
        for _,ticker,side,q,p,fee,_ in orders:
            sign=1 if side=="buy" else -1; positions[ticker]=positions.get(ticker,0)+sign*q
        return {"account_id":row[0],"cash":row[1],"initial_cash":row[2],"created_at":row[3],"positions":positions,"orders":[dict(zip(["id","ticker","side","quantity","price","fee","created_at"],o)) for o in orders]}
    def order(self,aid,ticker,side,quantity,price,fee):
        account=self.account(aid)
        if not account: raise ValueError("account not found")
        if side not in {"buy","sell"} or quantity<=0 or price<=0: raise ValueError("invalid order")
        held=account["positions"].get(ticker,0); gross=quantity*price; cost=gross*fee
        if side=="buy" and gross+cost>account["cash"]: raise ValueError("insufficient cash")
        if side=="sell" and quantity>held: raise ValueError("insufficient position")
        delta=-(gross+cost) if side=="buy" else gross-cost
        oid="order_"+uuid.uuid4().hex[:12]; now=datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute("update accounts set cash=cash+? where id=?",(delta,aid)); db.execute("insert into orders values(?,?,?,?,?,?,?)",(oid,aid,ticker,side,quantity,price,cost,now))
        return self.account(aid)
