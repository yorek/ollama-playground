# Chunking Samples

## Azure OpenAI

`sample-openai.py` uses Azure OpenAI to do Semantic Chunking. Make sure to add the Azure OpenAI endpoint and API key to the `.env` file.

## Local Models

`sample-local.py` uses Ollama to run local models. [Install Ollama locally](https://github.com/ollama/ollama) then make sure you have Ollama running

```bash
ollama serve
```

Then download the `mxbai-embed-large` model

```bash
ollama pull mxbai-embed-large
```

and then you can run the Python script.

To load and manage models it is also possible to use OpenWebUI

```
docker run -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```

Once the docker container is started, OpenWebUI will be available at

```
http://localhost:3000/
```

Otherwise Ollama can be completely managed via API, for example:

```
curl http://localhost:11434/api/tags | jq .
```

Documentation is here: https://github.com/ollama/ollama/tree/main/docs


## Azure

Ollama can also be deployed in an Azure container:

```bash
az containerapp env create -n OllamaAppEnv -g <resource-group> --location <location>
az containerapp create -n ollama -g <resource-group> --image ollama/ollama:latest --environment OllamaAppEnv --ingress external --target-port 11434 --cpu 4 --memory 8Gi 
```

Open Web UI can be connected to the deployed Ollama container with the folloowing script:

```bash
$fqdn=$(az containerapp ingress show -n ollama -g <resource-group> --query "fqdn" -o tsv)

docker run -p 3000:8080 -e OLLAMA_BASE_URL="https://$fqdn" -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```

