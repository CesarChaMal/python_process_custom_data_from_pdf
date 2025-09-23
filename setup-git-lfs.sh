#!/bin/bash
# Git LFS Setup Script for Large Files
# Usage: ./setup-git-lfs.sh

set -e

echo "🔧 Setting up Git LFS for large files..."

# Create .gitattributes if it doesn't exist
if [ ! -f .gitattributes ]; then
    echo "📝 Creating .gitattributes..."
    cat > .gitattributes << EOF
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.weights filter=lfs diff=lfs merge=lfs -text
EOF
else
    echo "✅ .gitattributes already exists"
fi

# Check if we have large files that need migration
LARGE_FILES=$(git ls-files | xargs -I {} sh -c 'if [ -f "{}" ] && [ $(stat -c%s "{}" 2>/dev/null || stat -f%z "{}" 2>/dev/null || echo 0) -gt 104857600 ]; then echo "{}"; fi' | head -5)

if [ -n "$LARGE_FILES" ]; then
    echo "🔍 Found large files that need LFS migration:"
    echo "$LARGE_FILES"
    
    echo "🚀 Migrating existing large files to LFS..."
    git add .gitattributes
    git lfs migrate import --include="*.safetensors,*.bin,*.pt,*.pth,*.onnx,*.h5,*.pkl,*.pickle,*.model,*.weights" --everything
    
    echo "📤 Force pushing with LFS..."
    git push --force-with-lease origin main
    
    echo "✅ Migration complete!"
else
    echo "✅ No large files found, just setting up LFS for future use"
    git add .gitattributes
    git commit -m "Add Git LFS configuration" || echo "No changes to commit"
    git push origin main || echo "Nothing to push"
fi

echo "🎉 Git LFS setup complete!"
echo "💡 Large files will now automatically use LFS"