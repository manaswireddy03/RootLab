# RootLab

RootLab is a Streamlit learning lab for Bisection and Newton-Raphson root finding. It includes an equation explainer, experiment history, and reference-root error statistics.

## Streamlit Community Cloud

Deploy the repository manually through Streamlit Community Cloud. Select `app.py` as the main file. The runtime dependencies are listed in the root-level `requirements.txt`; `rootlab_storage.py` is the local Python module imported by the app and must remain alongside `app.py`.

The app creates `rootlab_experiments.sqlite3` automatically in the application directory. The database is intentionally ignored by Git, so a deployment starts with an empty history. Community Cloud's local filesystem is ephemeral: records may disappear when the app is restarted, redeployed, or its runtime is replaced. SQLite here is suitable for demonstrations and short-lived experiments, not durable production history. Use an external managed database or persistent storage service if experiment records must survive runtime replacement.

Do not upload `.venv`, Python cache files, local secrets, or the local SQLite database. The source files, dependency manifest, and this README are sufficient to launch the app.

## Local checks

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```