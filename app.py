from flask import Flask, request, jsonify
from src.agents import run_research_report

app = Flask(__name__)

# Initialize the research agent system once at startup
agent_system = run_research_report()

@app.route("/")
def home():
    return jsonify({"message": "LangGraph Research Agent API is running."})


@app.route("/research", methods=["POST"])
def research():
    """
    POST JSON body:
    {
        "topic": "SK2小灯泡",
        "industry": "护肤"
    }
    """
    data = request.get_json(force=True)

    topic = data.get("topic")
    industry = data.get("industry")

    if not topic or not industry:
        return jsonify({
            "error": "Missing required fields. Both 'topic' and 'industry' are required."
        }), 400

    try:
        # Run the research agent
        result = agent_system.run(topic, industry)
        return jsonify({
            "topic": topic,
            "industry": industry,
            "result": result
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    # You can set host='0.0.0.0' to make it externally accessible
    app.run(host="0.0.0.0", port=5000, debug=True)
