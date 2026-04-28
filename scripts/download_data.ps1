$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "data/raw" | Out-Null

kaggle competitions download `
  -c planet-understanding-the-amazon-from-space `
  -p data/raw

Write-Host "Downloaded Kaggle archive(s) to data/raw."
Write-Host "Unzip the downloaded files so that train_v2.csv and train-jpg/ are under data/raw/."
