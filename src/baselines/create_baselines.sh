#!/bin/bash
set -e

# 0. Kill prior vLLM servers which may have been left running
pkill -f "python serve_models.py" || true

# 1. Serve Models
# Start the vLLM server in the background
python serve_models.py &
SERVER_PID=$!

# Wait for the server to be ready by checking the API endpoint
echo "Waiting for vLLM server to be ready..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 2
    echo "Still waiting for server..."
done
echo "Server is ready!"

# 2. Generate Outputs
python generate_outputs.py

# 3. Evaluate Outputs
python evaluate_judgments.py

# 4. Evaluate Prompts
python evaluate_prompts.py

# 5. Evaluate with StrongREJECT
python evaluate_responses_strongreject.py

# 6. Analyze Output Judgments
python analyze_judgments.py

# 7. Analyze Prompt Judgments
python analyze_prompt_judgments.py

# 8. Visualize Results
python visualize_judgments.py

# Kill the server
kill $SERVER_PID