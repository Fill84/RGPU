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

; User-defined variable for ProgramData path (NSIS has no $COMMONAPPDATA built-in)
Var ProgramDataDir

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
!define MUI_ICON "staging\icon.ico"
!define MUI_UNICON "staging\icon.ico"
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
; Init functions
;---------------------------------
Function .onInit
  ReadEnvStr $ProgramDataDir PROGRAMDATA
  StrCmp $ProgramDataDir "" 0 +2
    StrCpy $ProgramDataDir "C:\ProgramData"
FunctionEnd

Function un.onInit
  ReadEnvStr $ProgramDataDir PROGRAMDATA
  StrCmp $ProgramDataDir "" 0 +2
    StrCpy $ProgramDataDir "C:\ProgramData"
FunctionEnd

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

  ; Install binary and icon
  SetOutPath "$INSTDIR\bin"
  File "staging\rgpu.exe"
  File "staging\icon.ico"

  ; Create config directory and install default config
  CreateDirectory "$ProgramDataDir\RGPU"
  IfFileExists "$ProgramDataDir\RGPU\rgpu.toml" skip_config
    SetOutPath "$ProgramDataDir\RGPU"
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
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU" \
    "DisplayIcon" "$INSTDIR\bin\icon.ico"
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
  CreateShortCut "$SMPROGRAMS\RGPU\RGPU.lnk" "$INSTDIR\bin\rgpu.exe" "" "$INSTDIR\bin\icon.ico" 0
  CreateShortCut "$SMPROGRAMS\RGPU\Uninstall RGPU.lnk" "$INSTDIR\uninstall.exe"

  ; Desktop shortcut
  CreateShortCut "$DESKTOP\RGPU.lnk" "$INSTDIR\bin\rgpu.exe" "" "$INSTDIR\bin\icon.ico" 0
SectionEnd

;---------------------------------
; Section: CUDA System-Wide Interpose
;---------------------------------
Section "CUDA System-Wide Interpose (client only)" SEC_CUDA
  ; Install a copy to INSTDIR\lib for reference
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_cuda_interpose.dll"

  ; --- System-wide CUDA interception via System32 ---
  ; Back up the real NVIDIA nvcuda.dll if it exists
  IfFileExists "$SYSDIR\nvcuda.dll" 0 no_backup_needed
    ; Only backup if we haven't already (previous install)
    IfFileExists "$SYSDIR\nvcuda_real.dll" skip_backup
      DetailPrint "Backing up original nvcuda.dll to nvcuda_real.dll..."
      CopyFiles /SILENT "$SYSDIR\nvcuda.dll" "$SYSDIR\nvcuda_real.dll"
      WriteRegStr HKLM "Software\RGPU" "NvCudaBackedUp" "1"
    skip_backup:
  no_backup_needed:

  ; Copy our interpose DLL as nvcuda.dll into System32
  DetailPrint "Installing RGPU CUDA interpose as $SYSDIR\nvcuda.dll..."
  CopyFiles /SILENT "$INSTDIR\lib\rgpu_cuda_interpose.dll" "$SYSDIR\nvcuda.dll"
  WriteRegStr HKLM "Software\RGPU" "CudaInterposed" "1"
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
; Section: Server Windows Service
;---------------------------------
Section /o "Server Service (manual start)" SEC_SERVICE
  ; Create Windows Service using sc.exe
  ; Service starts manually (demand) -- user enables it explicitly
  nsExec::ExecToLog 'sc create "RGPU Server" binPath= "\"$INSTDIR\bin\rgpu.exe\" server --config \"$ProgramDataDir\RGPU\rgpu.toml\"" start= demand DisplayName= "RGPU Remote GPU Server"'
  nsExec::ExecToLog 'sc description "RGPU Server" "RGPU Remote GPU Sharing Server - exposes local GPUs over the network"'
SectionEnd

;---------------------------------
; Section: Client Daemon Service
;---------------------------------
Section /o "Client Daemon Service (auto-start)" SEC_CLIENT_SERVICE
  ; Create Windows Service for the client daemon with auto-start
  ; Reads config from the system-wide ProgramData location
  nsExec::ExecToLog 'sc create "RGPU Client" binPath= "\"$INSTDIR\bin\rgpu.exe\" client --config \"$ProgramDataDir\RGPU\rgpu.toml\"" start= auto DisplayName= "RGPU Client Daemon"'
  nsExec::ExecToLog 'sc description "RGPU Client" "RGPU Client Daemon - connects to remote GPU servers and exposes them locally via IPC"'

  ; Start the service immediately
  nsExec::ExecToLog 'sc start "RGPU Client"'
SectionEnd

;---------------------------------
; Component conflict warning
; (must be after all Section definitions so SEC_* identifiers are resolved)
;---------------------------------
Function .onSelChange
  ; Warn if both Server Service and CUDA Interpose are selected
  SectionGetFlags ${SEC_SERVICE} $0
  SectionGetFlags ${SEC_CUDA} $1
  IntOp $0 $0 & 1
  IntOp $1 $1 & 1
  IntCmp $0 0 no_conflict
  IntCmp $1 0 no_conflict
    MessageBox MB_YESNO|MB_ICONWARNING "WARNING: Installing CUDA interpose on a server machine will replace nvcuda.dll and break direct GPU access. Only install on CLIENT machines.$\r$\n$\r$\nKeep both selected?" IDYES no_conflict
    ; Deselect CUDA if user says No
    SectionGetFlags ${SEC_CUDA} $0
    IntOp $0 $0 & 0xFFFFFFFE
    SectionSetFlags ${SEC_CUDA} $0
  no_conflict:
FunctionEnd

;---------------------------------
; Section Descriptions
;---------------------------------
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CORE} \
    "The RGPU command-line tool and GUI. Required."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CUDA} \
    "System-wide CUDA interception. Replaces nvcuda.dll in System32 so ALL applications use remote GPUs. WARNING: Do NOT install on machines running the RGPU server."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_VULKAN} \
    "Vulkan Installable Client Driver (ICD) for presenting remote GPUs as local Vulkan devices."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_SERVICE} \
    "Install RGPU Server as a Windows Service (manual start). For machines with GPUs to share."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CLIENT_SERVICE} \
    "Install RGPU Client Daemon as an auto-start Windows Service. Connects to remote servers on boot. Reads config from $ProgramDataDir\RGPU\rgpu.toml."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;---------------------------------
; Uninstaller
;---------------------------------
Section "Uninstall"
  ; Stop and remove Windows Services (if installed)
  nsExec::ExecToLog 'sc stop "RGPU Client"'
  nsExec::ExecToLog 'sc delete "RGPU Client"'
  nsExec::ExecToLog 'sc stop "RGPU Server"'
  nsExec::ExecToLog 'sc delete "RGPU Server"'

  ; Restore original nvcuda.dll from backup (if we replaced it)
  ReadRegStr $0 HKLM "Software\RGPU" "CudaInterposed"
  StrCmp $0 "1" 0 skip_cuda_restore
    ; Check if backup exists
    IfFileExists "$SYSDIR\nvcuda_real.dll" 0 remove_our_nvcuda
      ; Restore the original NVIDIA driver
      DetailPrint "Restoring original nvcuda.dll from backup..."
      Delete "$SYSDIR\nvcuda.dll"
      CopyFiles /SILENT "$SYSDIR\nvcuda_real.dll" "$SYSDIR\nvcuda.dll"
      Delete "$SYSDIR\nvcuda_real.dll"
      Goto skip_cuda_restore
    remove_our_nvcuda:
      ; No backup existed (no NVIDIA driver was installed), just remove our DLL
      Delete "$SYSDIR\nvcuda.dll"
  skip_cuda_restore:

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
  Delete "$INSTDIR\bin\icon.ico"
  Delete "$INSTDIR\lib\rgpu_cuda_interpose.dll"
  Delete "$INSTDIR\lib\rgpu_vk_icd.dll"
  Delete "$INSTDIR\lib\rgpu_icd.json"
  Delete "$INSTDIR\uninstall.exe"

  ; Remove directories (only if empty)
  RMDir "$INSTDIR\bin"
  RMDir "$INSTDIR\lib"
  RMDir "$INSTDIR"

  ; Remove Start Menu and Desktop shortcuts
  Delete "$SMPROGRAMS\RGPU\RGPU.lnk"
  Delete "$SMPROGRAMS\RGPU\Uninstall RGPU.lnk"
  RMDir "$SMPROGRAMS\RGPU"
  Delete "$DESKTOP\RGPU.lnk"

  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\RGPU"
  DeleteRegKey HKLM "Software\RGPU"

  ; Note: Config in $ProgramDataDir\RGPU is intentionally NOT removed
  ; to preserve user configuration across reinstalls
SectionEnd
