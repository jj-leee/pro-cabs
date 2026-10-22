# Deploying PRO-CABS to https://pro-cabs.streamlit.app

Everything local is done and tested:

- the bundle is in `~/pro-cabs-streamlit`, with a local git repo on `main` (nothing pushed);
- `python smoke_test.py` passes;
- a headless `streamlit run app/base_app.py` launch works.

Nothing has been pushed. The steps below **publish** the app.

## 1. Create the GitHub repo and push

`gh` is authenticated as `jj-leee`, and the repo-local git identity uses the
noreply email.

GitHub allows only one fork of a repository per account. `jj-leee/pro-pann` is
already the fork of ML-Tongue-Pred, so `pro-cabs` is a standalone repository.
The README and `licenses/` credit ML-Tongue-Pred instead.

```bash
cd ~/pro-cabs-streamlit
git status --short     # should be clean
gh repo create pro-cabs --public --source . --remote origin --push
```

**Public** matches PRO-PANN. The repo holds code, trained models and aggregate
summaries only, with no patient-level data. Confirm your co-authors are fine
with publishing the trained model files before you push.

## 2. Deploy on share.streamlit.io

1. Go to https://share.streamlit.io and click **Create app**, then deploy from a GitHub repo.
2. Fill in the form:
   - **Repository:** `jj-leee/pro-cabs`
   - **Branch:** `main`
   - **Main file path:** `app/base_app.py`
   - **App URL:** `pro-cabs`, which gives **pro-cabs.streamlit.app**. If the subdomain is taken, pick another and update `URL` in `keep_awake.py`.
3. Click **Advanced settings** and set **Python version: 3.12**. This is required: the
   pickled models are pinned to Python 3.12 builds of the packages, and the version
   cannot be changed after deploy without deleting and redeploying.
4. Click **Deploy**. The first build takes a few minutes (`packages.txt` installs
   `libgomp1`; `requirements.txt` installs about 60 packages, with no torch).

## 3. Verify the live app

1. Open the app, tick **Augmentation (implant)** under Breast, keep the other
   defaults, and click **Estimate risk**.
2. Each of the four outcomes should show a risk, a tier, a percentile, an observed
   tier rate and its own contributing-factors chart.
3. Compare with the local reference values that `python smoke_test.py` prints for
   the "breast (augmentation)" profile.
4. If anything fails, open **Manage app** (bottom right) and check the logs.

## Notes

- The keep-awake workflow (`.github/workflows/keep_awake.yml`) pings the app every
  8 hours. It fails until the app is live, so expect red runs before step 2.
- To update the models, rerun the private pipeline in `ML-Cosmetic-Pred/`, then
  `python tools/build_deidentified_artifacts.py --orig "<path>/ML-Cosmetic-Pred/app"`,
  rerun `smoke_test.py`, commit and push. Never copy `app/all_preds/` or any data
  files into this repo.
- After pushing changes to imported modules, use **Reboot app** on
  share.streamlit.io, because the running app can keep the old module in memory.
