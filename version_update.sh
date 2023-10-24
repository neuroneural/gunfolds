python_file1="version.py"

version=$(grep "__version_info__ =" "$python_file1" | sed 's/.*\[\(.*\)\].*/\1/')
sed -i -e "1s/s = .*/s = [$version]/" version_update.py
git add version_update.py
git commit -m "version update"
git push origin master

updated_version=$(python version_update.py)
sed -i -e "1s/__version_info__ = .*/__version_info__ = $updated_version/" version.py
git add version.py
git commit -m "version update"
git push origin master
rm *.py-e