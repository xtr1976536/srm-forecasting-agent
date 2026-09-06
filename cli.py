import argparse,json
from .agent import SRMForecastingAgent
def main():
    p=argparse.ArgumentParser(); p.add_argument("request"); p.add_argument("--csv"); p.add_argument("--out",default="srm_agent_runs"); a=p.parse_args(); agent=SRMForecastingAgent(a.out); result=agent.run(a.request,a.csv); print(json.dumps(result,indent=2)); print(agent.explain(result,next(iter(result["predictions"]))))
if __name__=="__main__": main()
