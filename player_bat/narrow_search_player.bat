@echo off
cd /d %~dp0
cd ..
python -m neneshogi.narrow_search_player
