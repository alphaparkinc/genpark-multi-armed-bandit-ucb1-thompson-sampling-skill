"""MCP Server for Multi-Armed Bandit Skill."""
import json
import sys
from client import MultiArmedBandit

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/list":
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [{
                            "name": "run_bandit_selection",
                            "description": "Select best arm using UCB1 algorithm",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "num_arms": {"type": "integer"},
                                    "history": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "arm": {"type": "integer"},
                                                "reward": {"type": "number"}
                                            },
                                            "required": ["arm", "reward"]
                                        }
                                    },
                                    "t": {"type": "integer"}
                                },
                                "required": ["num_arms", "t"]
                            }
                        }]
                    }
                }
            elif method == "tools/call":
                args = params.get("arguments", {})
                mab = MultiArmedBandit(args["num_arms"])
                for h in args.get("history", []):
                    mab.update(h["arm"], h["reward"])
                selected = mab.select_arm_ucb1(args["t"])
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps({"selected_arm": selected, "summary": mab.get_summary()})}]}
                }
            else:
                res = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
            print(json.dumps(res), flush=True)
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(e)}}
            print(json.dumps(err), flush=True)

if __name__ == "__main__":
    main()
