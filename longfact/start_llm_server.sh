export CUDA_VISIBLE_DEVICES=0,1,2,5
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
python3 -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --served-model-name llama3.1 \
        --tensor-parallel-size 4 \
        --trust-remote-code
        
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.1",
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ]
    }'