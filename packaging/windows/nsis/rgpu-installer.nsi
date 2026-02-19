; RGPU Installer Script for NSIS
; Produces: rgpu-VERSION-windows-x64-setup.exe
;
; Prerequisites:
;   - NSIS 3.x installed (winget install NSIS.NSIS)
;   - Release build artifacts in staging/ subdirectory
;
; Build: makensis /DVERSION=0.1.0 rgpu-installer.nsi

;---------------------------------
; Includes
;---------------------------------
!include "MUI2.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"
!include "WordFunc.nsh"
!include "WinMessages.nsh"

;---------------------------------
; General
;---------------------------------
!ifndef VERSION
  !define VERSION "0.1.0"
!endif

Name "RGPU ${VERSION}"
OutFile "rgpu-${VERSION}-windows-x64-setup.exe"
InstallDir "$PROGRAMFILES64\RGPU"
InstallDirRegKey HKLM "Software\RGPU" "InstallDir"
RequestExecutionLevel admin
Unicode True

;---------------------------------
; Version Info
;---------------------------------
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "RGPU"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileDescription" "RGPU Remote GPU Sharing Installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "MIT OR Apache-2.0"

;---------------------------------
; Interface Settings
;---------------------------------
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "Welcome to RGPU ${VERSION} Setup"
!define MUI_WELCOMEPAGE_TEXT "This wizard will install RGPU (Remote GPU Sharing) on your computer.$\r$\n$\r$\nRGPU enables sharing GPUs over the network, supporting both Vulkan and CUDA workloads.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_TITLE "RGPU ${VERSION} Installation Complete"
!define MUI_FINISHPAGE_TEXT "RGPU has been installed on your computer.$\r$\n$\r$\nQuick start:$\r$\n  Server: rgpu server$\r$\n  Client: rgpu client --server host:9876$\r$\n  UI:     rgpu ui$\r$\n$\r$\nClick Finish to close this wizard."

;---------------------------------
; Pages
;---------------------------------
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "license.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

;---------------------------------
; Section: Core (required)
;---------------------------------
Section "RGPU Core (required)" SEC_CORE
  SectionIn RO

  ; Check 64-bit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "RGPU requires a 64-bit version of Windows."
    Abort
  ${EndIf}

  ; Install binary
  SetOutPath "$INSTDIR\bin"
  File "staging\rgpu.exe"

  ; Create config directory and install default config
  CreateDirectory "$COMMONAPPDATA\RGPU"
  IfFileExists "$COMMONAPPDATA\RGPU\rgpu.toml" skip_config
    SetOutPath "$COMMONAPPDATA\RGPU"
    File "/oname=rgpu.toml" "staging\rgpu.toml.template"
  skip_config:

  ; Add to system PATH (native registry manipulation)
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path"
  StrCpy $0 "$0;$INSTDIR\bin"
  WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path" $0
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000

  ; Store installation directory
  WriteRegStr HKLM "Software\RGPU" "InstallDir" "$INSTDIR"

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; Add/Remove Programs entry
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "DisplayName" "RGPU ${VERSION} - Remote GPU Sharing"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "Publisher" "RGPU Project"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "NoRepair" 1

  ; Calculate installed size
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "EstimatedSize" $0

  ; Start Menu shortcuts
  CreateDirectory "$SMPROGRAMS\RGPU"
  CreateShortCut "$SMPROGRAMS\RGPU\Uninstall RGPU.lnk" "$INSTDIR\uninstall.exe"
SectionEnd

;---------------------------------
; Section: CUDA Interpose Library
;---------------------------------
Section "CUDA Interpose Library" SEC_CUDA
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_cuda_interpose.dll"
SectionEnd

;---------------------------------
; Section: Vulkan ICD Driver
;---------------------------------
Section "Vulkan ICD Driver" SEC_VULKAN
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_vk_icd.dll"

  ; Generate ICD manifest with correct absolute path
  ; Use forward slashes for Vulkan loader compatibility
  FileOpen $0 "$INSTDIR\lib\rgpu_icd.json" w
  FileWrite $0 '{$\r$\n'
  FileWrite $0 '    "file_format_version": "1.0.1",$\r$\n'
  FileWrite $0 '    "ICD": {$\r$\n'
  FileWrite $0 '        "library_path": "$INSTDIR\lib\rgpu_vk_icd.dll",$\r$\n'
  FileWrite $0 '        "api_version": "1.3.0",$\r$\n'
  FileWrite $0 '        "is_portability_driver": false$\r$\n'
  FileWrite $0 '    }$\r$\n'
  FileWrite $0 '}$\r$\n'
  FileClose $0

  ; Register Vulkan ICD in Windows registry
  ; The Vulkan loader checks HKLM\SOFTWARE\Khronos\Vulkan\Drivers for ICD manifests
  WriteRegDWORD HKLM "SOFTWARE\Khronos\Vulkan\Drivers" \
    "$INSTDIR\lib\rgpu_icd.json" 0
SectionEnd

;---------------------------------
; Section: Windows Service
;---------------------------------
Section /o "Windows Service (manual start)" SEC_SERVICE
  ; Create Windows Service using sc.exe
  ; Service starts manually (demand) -- user enables it explicitly
  nsExec::ExecToLog 'sc create "RGPU Server" binPath= "\"$INSTDIR\bin\rgpu.exe\" server --config \"$COMMONAPPDATA\RGPU\rgpu.toml\"" start= demand DisplayName= "RGPU Remote GPU Server"'
  nsExec::ExecToLog 'sc description "RGPU Server" "RGPU Remote GPU Sharing Server - exposes local GPUs over the network"'
SectionEnd

;---------------------------------
; Section Descriptions
;---------------------------------
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CORE} \
    "The RGPU command-line tool and GUI. Required."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CUDA} \
    "CUDA interpose library for intercepting CUDA API calls and forwarding them to remote GPUs."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_VULKAN} \
    "Vulkan Installable Client Driver (ICD) for presenting remote GPUs as local Vulkan devices."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_SERVICE} \
    "Install RGPU as a Windows Service (manual start). Use 'sc start RGPU Server' to run."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;---------------------------------
; Uninstaller
;---------------------------------
Section "Uninstall"
  ; Stop and remove Windows Service (if installed)
  nsExec::ExecToLog 'sc stop "RGPU Server"'
  nsExec::ExecToLog 'sc delete "RGPU Server"'

  ; Remove Vulkan ICD registry entry
  DeleteRegValue HKLM "SOFTWARE\Khronos\Vulkan\Drivers" "$INSTDIR\lib\rgpu_icd.json"

  ; Remove from PATH (native registry manipulation)
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path"
  ${WordReplace} $0 ";$INSTDIR\bin" "" "+" $0
  ${WordReplace} $0 "$INSTDIR\bin;" "" "+" $0
  ${WordReplace} $0 "$INSTDIR\bin" "" "+" $0
  WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path" $0
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000

  ; Remove files
  Delete "$INSTDIR\bin\rgpu.exe"
  Delete "$INSTDIR\lib\rgpu_cuda_interpose.dll"
  Delete "$INSTDIR\lib\rgpu_vk_icd.dll"
  Delete "$INSTDIR\lib\rgpu_icd.json"
  Delete "$INSTDIR\uninstall.exe"

  ; Remove directories (only if empty)
  RMDir "$INSTDIR\bin"
  RMDir "$INSTDIR\lib"
  RMDir "$INSTDIR"

  ; Remove Start Menu shortcuts
  Delete "$SMPROGRAMS\RGPU\Uninstall RGPU.lnk"
  RMDir "$SMPROGRAMS\RGPU"

  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU"
  DeleteRegKey HKLM "Software\RGPU"

  ; Note: Config in $COMMONAPPDATA\RGPU is intentionally NOT removed
  ; to preserve user configuration across reinstalls
SectionEnd
