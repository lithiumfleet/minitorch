mkdir .env
python -m venv .env
source .env/bin/activate
echo "\n.env/" >> .gitignore

pip install -r ./requirements.extra.txt
pip install -r ./requirements.txt