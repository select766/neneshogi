@echo off
cd /d %~dp0
cd ..
python -m neneshogi.neneshogi ProbSearchPlayer
