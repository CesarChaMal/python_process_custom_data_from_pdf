@echo off
REM Git LFS Setup Script for Large Files (Windows)
REM Usage: setup-git-lfs.bat

echo 🔧 Setting up Git LFS for large files...

REM Create .gitattributes if it doesn't exist
if not exist .gitattributes (
    echo 📝 Creating .gitattributes...
    (
        echo *.safetensors filter=lfs diff=lfs merge=lfs -text
        echo *.bin filter=lfs diff=lfs merge=lfs -text
        echo *.pt filter=lfs diff=lfs merge=lfs -text
        echo *.pth filter=lfs diff=lfs merge=lfs -text
        echo *.onnx filter=lfs diff=lfs merge=lfs -text
        echo *.h5 filter=lfs diff=lfs merge=lfs -text
        echo *.pkl filter=lfs diff=lfs merge=lfs -text
        echo *.pickle filter=lfs diff=lfs merge=lfs -text
        echo *.model filter=lfs diff=lfs merge=lfs -text
        echo *.weights filter=lfs diff=lfs merge=lfs -text
    ) > .gitattributes
) else (
    echo ✅ .gitattributes already exists
)

echo 🚀 Migrating existing large files to LFS...
git add .gitattributes
git lfs migrate import --include="*.safetensors,*.bin,*.pt,*.pth,*.onnx,*.h5,*.pkl,*.pickle,*.model,*.weights" --everything

echo 📤 Force pushing with LFS...
git push --force-with-lease origin main

echo 🎉 Git LFS setup complete!
echo 💡 Large files will now automatically use LFS