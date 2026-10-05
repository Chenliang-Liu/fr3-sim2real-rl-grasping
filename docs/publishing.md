# Publish the prepared repository

The intended destination is `Chenliang-Liu/fr3-sim2real-rl-grasping`, public visibility. This file describes publication; its presence is not proof that a GitHub repository exists.

The prepared local checkout already has a `main` branch and an `origin` URL. Keep the code, media, thesis, checkpoint, and log directory structure intact. All included files are below GitHub's [100 MiB per-file Git limit](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository), so this archive does not require Git LFS.

If you extracted the distribution ZIP instead of using the prepared checkout, the ZIP excludes Git metadata. Initialize the extracted directory first:

```bash
git init -b main
git config user.name "Chenliang Liu"
git config user.email "Chenliang-Liu@users.noreply.github.com"
git add .
git commit -m "Publish undergraduate FR3 Sim2Real research archive"
git remote add origin https://github.com/Chenliang-Liu/fr3-sim2real-rl-grasping.git
```

## With GitHub CLI

With Git and the [official GitHub CLI](https://cli.github.com/) installed, sign in as the intended account yourself:

```bash
gh auth login
gh api user --jq .login
gh auth setup-git
```

The reported login should be `Chenliang-Liu`. From the repository root, after reviewing the README and included materials:

```bash
gh repo create Chenliang-Liu/fr3-sim2real-rl-grasping \
  --public \
  --description "Undergraduate research: SAC cube grasping in MuJoCo/robosuite and preliminary Sim2Real deployment on Franka Research 3"
git push -u origin main
```

The checkout already has an `origin` remote, used by the push command. If a repository of this name already exists, inspect it before publishing instead of replacing its history. This guide does not authorize overwriting another repository.

## With GitHub's website and Git

Create an empty public repository named `fr3-sim2real-rl-grasping` under `Chenliang-Liu`. Leave the new repository's README, `.gitignore`, and license initialization choices unchecked because this checkout already supplies its own files. Then, from the checkout:

```bash
git push -u origin main
```

Use GitHub's normal authentication flow for Git. Do not put a personal access token in a repository file, remote URL, or public terminal log.

After publication, open the repository and verify the README previews, videos, thesis, and file tree. Suggested repository topics are `reinforcement-learning`, `sim2real`, `franka`, `robotic-manipulation`, `mujoco`, `robosuite`, and `soft-actor-critic`.
