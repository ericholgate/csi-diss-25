# Git Setup for Personal GitHub Repository

## Configure Git for This Repository

To use your personal GitHub account instead of your default work account for this specific repository:

### Option 1: Repository-Specific Configuration (Recommended)
```bash
cd /Users/eholgate/research/csi-diss-25

# Set your personal GitHub credentials for this repo only
git config user.name "Your Personal Name"
git config user.email "your_personal_email@gmail.com"

# Verify the configuration
git config user.name
git config user.email
```

### Option 2: Use SSH with Different Keys
If you have separate SSH keys for personal vs work:

```bash
# Check existing SSH keys
ls -la ~/.ssh/

# If you need to generate a personal key
ssh-keygen -t ed25519 -C "your_personal_email@gmail.com" -f ~/.ssh/id_personal_github

# Add key to SSH agent
ssh-add ~/.ssh/id_personal_github

# Configure SSH to use the right key for GitHub
# Add to ~/.ssh/config:
Host github.com-personal
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_personal_github

# Then use this remote URL format:
git remote add origin git@github.com-personal:your_username/csi-diss-25.git
```

### Option 3: Use Personal Access Token
```bash
# Create a Personal Access Token in GitHub Settings > Developer settings
# Then use HTTPS with token authentication:
git remote add origin https://your_username:your_token@github.com/your_username/csi-diss-25.git
```

## Initialize and Push Repository

```bash
# Initialize if not already a git repo
git init

# Check what files will be committed (should exclude notes/, scratch/, etc.)
git status

# Add files (gitignore will handle exclusions)
git add .

# Make initial commit
git commit -m "Initial commit: CSI character embedding learning system

🤖 Generated with Claude Code (https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Create repository on GitHub first (private), then:
git remote add origin git@github.com:your_username/csi-diss-25.git

# Push to GitHub
git push -u origin main
```

## Verify Configuration
```bash
# Check current git configuration for this repo
git config --list --local

# Verify remote repository
git remote -v

# Check that notes/ and scratch/ are ignored
git status --ignored
```

## Troubleshooting

### If you get "Permission denied"
- Check SSH key is added: `ssh -T git@github.com`
- Verify key permissions: `chmod 600 ~/.ssh/id_*`
- Add key to GitHub: Settings > SSH and GPG keys

### If commits show wrong author
- Check local config: `git config user.email`
- Amend last commit: `git commit --amend --author="Your Name <email@example.com>"`

### If gitignore not working
- Clear git cache: `git rm -r --cached .`
- Re-add files: `git add .`
- Commit: `git commit -m "Update gitignore"`