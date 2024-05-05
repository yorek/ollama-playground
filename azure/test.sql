declare @response nvarchar(max);
exec sp_invoke_external_rest_endpoint 
    @url='https://ollama.ashyriver-12176fdd.northcentralusstage.azurecontainerapps.io/api/embeddings',
    @method='POST',
    @headers='{"Content-Type": "application/json"}',
    @payload='{
        "model": "mxbai-embed-large",
        "prompt": "Llamas are members of the camelid family"
    }',
    @response=@response output
select @response