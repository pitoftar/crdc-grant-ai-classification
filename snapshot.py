from huggingface_hub import snapshot_download

snapshot_download(repo_id='facebook/bart-large-mnli', repo_type='model') # ajuster le/les modeles pour les tests hors ligne