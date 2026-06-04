@echo off
for %%f in (*.eps) do (
    epstopdf "%%f"
    echo Converted: %%f
)
echo All done!
pause