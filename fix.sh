cd .github/workflows
sed -i \
  -e 's#actions/checkout@v4#actions/checkout@v5#g' \
  -e 's#actions/setup-python@v5#actions/setup-python@v6#g' \
  -e 's#actions/upload-artifact@v4#actions/upload-artifact@v6#g' \
  *.yml
git add -A && git commit -m "Bump actions to Node 24 versions" && git push
