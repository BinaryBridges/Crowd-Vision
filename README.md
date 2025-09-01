<div align="center">

<pre>
██████╗░░█████╗░██████╗░░█████╗░██╗░░██╗  ███████╗██╗░░░██╗███████╗
██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║░░██║  ██╔════╝╚██╗░██╔╝██╔════╝
██████╔╝██║░░██║██████╔╝██║░░╚═╝███████║  █████╗░░░╚████╔╝░█████╗░░
██╔═══╝░██║░░██║██╔══██╗██║░░██╗██╔══██║  ██╔══╝░░░░╚██╔╝░░██╔══╝░░
██║░░░░░╚█████╔╝██║░░██║╚█████╔╝██║░░██║  ███████╗░░░██║░░░███████╗
╚═╝░░░░░░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝  ╚══════╝░░░╚═╝░░░╚══════╝
</pre>

</div>

### Steps to run

1. Start virtual environment
`py -m venv .venv` 
`.\.venv\Scripts\Activate`

2. Install requirements just for running the app
`pip install -r requirements.txt`

for developer
'pip install -r requirements-dev.txt'

3. Start project
`python main.py`

### Formatting and linting
ruff format .
ruff check --fix .