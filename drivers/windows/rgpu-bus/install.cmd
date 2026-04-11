@echo off
REM install.cmd — Install rgpu-bus driver with test signing
REM Run as Administrator!

echo ============================================
echo  RGPU Bus Driver Installer (test-signed)
echo ============================================
echo.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Run this as Administrator!
    pause
    exit /b 1
)

bcdedit /set testsigning on >nul 2>&1

set DRIVER_DIR=%~dp0x64\Debug
if not exist "%DRIVER_DIR%\rgpu-bus.sys" (
    echo ERROR: Build the driver first (msbuild)
    pause
    exit /b 1
)

makecert -r -pe -ss PrivateCertStore -n "CN=RGPU Test" RGPU_Test.cer >nul 2>&1
signtool sign /s PrivateCertStore /n "RGPU Test" /t http://timestamp.digicert.com /fd SHA256 "%DRIVER_DIR%\rgpu-bus.sys"

copy /y rgpu-bus.inf "%DRIVER_DIR%\" >nul

echo Installing driver...
devcon install "%DRIVER_DIR%\rgpu-bus.inf" Root\RGPUBus

echo.
echo Done! Reboot if test-signing was just enabled.
pause
