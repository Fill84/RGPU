@echo off
REM ============================================================================
REM  build.cmd - Build RGPU Virtual GPU Driver
REM
REM  Prerequisites:
REM    1. Visual Studio 2022 (Community or higher) with "Desktop development
REM       with C++" workload installed
REM    2. Windows Driver Kit (WDK) 10.0.26100.x or later
REM       Download: https://learn.microsoft.com/en-us/windows-hardware/drivers/download-the-wdk
REM    3. Windows SDK (matching WDK version)
REM
REM  Usage:
REM    build.cmd [debug|release]
REM
REM  Output:
REM    build\<config>\rgpu-vgpu.sys   - Driver binary
REM    rgpu-vgpu.inf                  - INF file (in source directory)
REM    rgpu-vgpu.cat                  - Catalog file (created by signing)
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM --- Parse arguments ---
set CONFIG=Release
if /I "%1"=="debug" set CONFIG=Debug
if /I "%1"=="release" set CONFIG=Release

set ARCH=x64
set TARGETDIR=%~dp0build\%CONFIG%

echo.
echo ================================================================
echo  RGPU Virtual GPU Driver Build
echo  Configuration: %CONFIG%
echo  Architecture:  %ARCH%
echo ================================================================
echo.

REM --- Check for WDK installation ---
REM Try to find the WDK via common installation paths
set WDK_ROOT=
for %%P in (
    "C:\Program Files (x86)\Windows Kits\10"
    "C:\Program Files\Windows Kits\10"
) do (
    if exist "%%~P\Include" (
        set "WDK_ROOT=%%~P"
    )
)

if "%WDK_ROOT%"=="" (
    echo [ERROR] Windows Driver Kit (WDK) not found!
    echo.
    echo Please install the following:
    echo   1. Visual Studio 2022 with "Desktop development with C++" workload
    echo   2. Windows SDK from https://developer.microsoft.com/windows/downloads/windows-sdk/
    echo   3. Windows WDK from https://learn.microsoft.com/en-us/windows-hardware/drivers/download-the-wdk
    echo.
    echo Make sure the WDK VS extension is also installed.
    exit /b 1
)

echo [OK] WDK found at: %WDK_ROOT%

REM --- Find the latest WDK version ---
set WDK_VERSION=
for /d %%V in ("%WDK_ROOT%\Include\*") do (
    set "WDK_VERSION=%%~nxV"
)

if "%WDK_VERSION%"=="" (
    echo [ERROR] Could not determine WDK version from Include directory.
    exit /b 1
)

echo [OK] WDK version: %WDK_VERSION%

REM --- Locate MSBuild ---
set MSBUILD=
for %%M in (
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
) do (
    if exist "%%~M" (
        set "MSBUILD=%%~M"
    )
)

REM --- Alternative: try to find via vswhere ---
if "%MSBUILD%"=="" (
    for /f "usebackq tokens=*" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe 2^>nul`) do (
        set "MSBUILD=%%I"
    )
)

if "%MSBUILD%"=="" (
    echo [WARNING] MSBuild not found. Falling back to direct cl.exe compilation.
    echo          For best results, open a "Developer Command Prompt for VS 2022"
    echo          or install Visual Studio with the C++ workload.
    goto :ManualBuild
)

echo [OK] MSBuild found at: %MSBUILD%

REM --- Check if .vcxproj exists, if not create a minimal one ---
if not exist "%~dp0rgpu-vgpu.vcxproj" (
    echo [INFO] No .vcxproj found. Generating a minimal project file...
    call :GenerateVcxproj
)

REM --- Build with MSBuild ---
echo.
echo [BUILD] Building with MSBuild...
echo.
"%MSBUILD%" "%~dp0rgpu-vgpu.vcxproj" /p:Configuration=%CONFIG% /p:Platform=%ARCH% /v:m

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed! See errors above.
    exit /b 1
)

echo.
echo [OK] Build succeeded!
echo     Output: %TARGETDIR%\rgpu-vgpu.sys
goto :SignInfo

REM ============================================================================
:ManualBuild
REM Fallback: use cl.exe directly (requires WDK environment to be set up)
REM ============================================================================

where cl.exe >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] cl.exe not found in PATH.
    echo         Please run this from a "Developer Command Prompt for VS 2022"
    echo         or install Visual Studio with the C++ workload.
    exit /b 1
)

echo.
echo [BUILD] Direct compilation with cl.exe...
echo         (This is a simplified build; use MSBuild for production.)
echo.

if not exist "%TARGETDIR%" mkdir "%TARGETDIR%"

set "KMDF_INC=%WDK_ROOT%\Include\wdf\kmdf\1.33"
set "WDK_INC=%WDK_ROOT%\Include\%WDK_VERSION%\km"
set "WDK_SHARED=%WDK_ROOT%\Include\%WDK_VERSION%\shared"
set "WDK_LIB=%WDK_ROOT%\Lib\%WDK_VERSION%\km\x64"
set "KMDF_LIB=%WDK_ROOT%\Lib\wdf\kmdf\x64\1.33"

cl.exe /nologo /kernel /W4 /WX /D_KERNEL_MODE /DNTDDI_VERSION=0x0A00000B ^
    /I"%WDK_INC%" /I"%WDK_SHARED%" /I"%KMDF_INC%" ^
    /Fo"%TARGETDIR%\rgpu-vgpu.obj" /c "%~dp0rgpu-vgpu.c"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed.
    exit /b 1
)

link.exe /nologo /kernel /driver:wdm /subsystem:native /entry:DriverEntry ^
    /out:"%TARGETDIR%\rgpu-vgpu.sys" ^
    "%TARGETDIR%\rgpu-vgpu.obj" ^
    "%WDK_LIB%\ntoskrnl.lib" "%WDK_LIB%\hal.lib" "%WDK_LIB%\wmilib.lib" ^
    "%KMDF_LIB%\wdfldr.lib" "%WDK_LIB%\ntstrsafe.lib"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Linking failed.
    exit /b 1
)

echo.
echo [OK] Build succeeded!
echo     Output: %TARGETDIR%\rgpu-vgpu.sys

REM ============================================================================
:SignInfo
REM ============================================================================
echo.
echo ================================================================
echo  Next Steps:
echo ================================================================
echo.
echo  1. TEST SIGNING (development):
echo     signtool sign /v /s PrivateCertStore /n "YourTestCert" ^
echo         /t http://timestamp.digicert.com "%TARGETDIR%\rgpu-vgpu.sys"
echo     inf2cat /driver:"%~dp0" /os:10_X64
echo     signtool sign /v /s PrivateCertStore /n "YourTestCert" ^
echo         /t http://timestamp.digicert.com "%~dp0rgpu-vgpu.cat"
echo.
echo  2. INSTALL:
echo     pnputil /add-driver "%~dp0rgpu-vgpu.inf" /install
echo.
echo  3. CREATE DEVICE INSTANCE:
echo     devcon install "%~dp0rgpu-vgpu.inf" Root\RGPU_VGPU
echo     (or use the Rust device_manager SetupDi APIs)
echo.
echo  Remember: Enable test signing with:
echo     bcdedit /set testsigning on
echo ================================================================

exit /b 0

REM ============================================================================
:GenerateVcxproj
REM Generate a minimal .vcxproj for WDK/KMDF driver build
REM ============================================================================

(
echo ^<?xml version="1.0" encoding="utf-8"?^>
echo ^<Project DefaultTargets="Build" ToolsVersion="Current" xmlns="http://schemas.microsoft.com/developer/msbuild/2003"^>
echo   ^<ItemGroup Label="ProjectConfigurations"^>
echo     ^<ProjectConfiguration Include="Debug^|x64"^>
echo       ^<Configuration^>Debug^</Configuration^>
echo       ^<Platform^>x64^</Platform^>
echo     ^</ProjectConfiguration^>
echo     ^<ProjectConfiguration Include="Release^|x64"^>
echo       ^<Configuration^>Release^</Configuration^>
echo       ^<Platform^>x64^</Platform^>
echo     ^</ProjectConfiguration^>
echo   ^</ItemGroup^>
echo   ^<PropertyGroup Label="Globals"^>
echo     ^<ProjectGuid^>{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}^</ProjectGuid^>
echo     ^<TemplateGuid^>{1bc93793-694f-48fe-9372-81e2b05556fd}^</TemplateGuid^>
echo     ^<TargetFrameworkVersion^>v4.5^</TargetFrameworkVersion^>
echo     ^<MinimumVisualStudioVersion^>12.0^</MinimumVisualStudioVersion^>
echo     ^<Configuration^>Debug^</Configuration^>
echo     ^<Platform Condition="'$(Platform)' == ''"^>x64^</Platform^>
echo     ^<RootNamespace^>rgpu_vgpu^</RootNamespace^>
echo     ^<DriverType^>KMDF^</DriverType^>
echo     ^<DriverTargetPlatform^>Universal^</DriverTargetPlatform^>
echo   ^</PropertyGroup^>
echo   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" /^>
echo   ^<PropertyGroup Label="Configuration" Condition="'$(Configuration)^|$(Platform)'=='Debug^|x64'"^>
echo     ^<TargetVersion^>Windows10^</TargetVersion^>
echo     ^<UseDebugLibraries^>true^</UseDebugLibraries^>
echo     ^<PlatformToolset^>WindowsKernelModeDriver10.0^</PlatformToolset^>
echo     ^<ConfigurationType^>Driver^</ConfigurationType^>
echo     ^<KmdfVersionMajor^>1^</KmdfVersionMajor^>
echo     ^<KmdfVersionMinor^>33^</KmdfVersionMinor^>
echo   ^</PropertyGroup^>
echo   ^<PropertyGroup Label="Configuration" Condition="'$(Configuration)^|$(Platform)'=='Release^|x64'"^>
echo     ^<TargetVersion^>Windows10^</TargetVersion^>
echo     ^<UseDebugLibraries^>false^</UseDebugLibraries^>
echo     ^<PlatformToolset^>WindowsKernelModeDriver10.0^</PlatformToolset^>
echo     ^<ConfigurationType^>Driver^</ConfigurationType^>
echo     ^<KmdfVersionMajor^>1^</KmdfVersionMajor^>
echo     ^<KmdfVersionMinor^>33^</KmdfVersionMinor^>
echo   ^</PropertyGroup^>
echo   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" /^>
echo   ^<ItemGroup^>
echo     ^<ClCompile Include="rgpu-vgpu.c" /^>
echo   ^</ItemGroup^>
echo   ^<ItemGroup^>
echo     ^<Inf Include="rgpu-vgpu.inf" /^>
echo   ^</ItemGroup^>
echo   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" /^>
echo   ^<Import Project="$(VCTargetsPath)\Microsoft.Cpp.Wdk.targets" Condition="Exists('$(VCTargetsPath)\Microsoft.Cpp.Wdk.targets')" /^>
echo ^</Project^>
) > "%~dp0rgpu-vgpu.vcxproj"

echo [OK] Generated rgpu-vgpu.vcxproj
goto :eof
