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

; Finish page: if reboot is needed (locked NVML), offer reboot option
!define MUI_FINISHPAGE_REBOOTLATER_DEFAULT
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

  ; Support /INSTALLTYPE=client or /INSTALLTYPE=server for silent installs
  ; Usage: setup.exe /S /INSTALLTYPE=server
  ; Section indices: 0=Core(RO), 1=Client, 2=Server
  ${GetParameters} $0
  ${GetOptions} $0 "/INSTALLTYPE=" $1
  StrCmp $1 "" init_done
  StrCmp $1 "server" select_server
  StrCmp $1 "Server" select_server
  ; Default: client (already selected by default)
  Goto init_done

  select_server:
    ; Deselect Client (section 1), select Server (section 2)
    SectionGetFlags 1 $2
    IntOp $2 $2 & 0xFFFFFFFE
    SectionSetFlags 1 $2
    SectionGetFlags 2 $2
    IntOp $2 $2 | 1
    SectionSetFlags 2 $2

  init_done:
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
  ; Grant Users group Modify access so UI can save config without elevation
  nsExec::ExecToLog 'icacls "$ProgramDataDir\RGPU" /grant Users:(OI)(CI)M /T'
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
; Installation Type: Client
; Installs all interpose DLLs (CUDA, NVENC, NVDEC, NVML, NVAPI),
; Vulkan ICD, and RGPU Client Daemon service (auto-start with OS).
;---------------------------------
Section "Client Installation" SEC_CLIENT
  ; ---- CUDA Interpose ----
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_cuda_interpose.dll"

  ${DisableX64FSRedirection}

  IfFileExists "$SYSDIR\nvcuda.dll" 0 no_cuda_backup_needed
    IfFileExists "$SYSDIR\nvcuda_real.dll" skip_cuda_backup
      DetailPrint "Backing up original nvcuda.dll to nvcuda_real.dll..."
      CopyFiles /SILENT "$SYSDIR\nvcuda.dll" "$SYSDIR\nvcuda_real.dll"
      WriteRegStr HKLM "Software\RGPU" "NvCudaBackedUp" "1"
    skip_cuda_backup:
  no_cuda_backup_needed:

  DetailPrint "Installing RGPU CUDA interpose as $SYSDIR\nvcuda.dll..."
  CopyFiles /SILENT "$INSTDIR\lib\rgpu_cuda_interpose.dll" "$SYSDIR\nvcuda.dll"
  WriteRegStr HKLM "Software\RGPU" "CudaInterposed" "1"

  ${EnableX64FSRedirection}

  ; ---- NVENC Interpose ----
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_nvenc_interpose.dll"

  ${DisableX64FSRedirection}

  IfFileExists "$SYSDIR\nvEncodeAPI64.dll" 0 no_nvenc_backup_needed
    IfFileExists "$SYSDIR\nvEncodeAPI64_real.dll" skip_nvenc_backup
      DetailPrint "Backing up original nvEncodeAPI64.dll..."
      CopyFiles /SILENT "$SYSDIR\nvEncodeAPI64.dll" "$SYSDIR\nvEncodeAPI64_real.dll"
      WriteRegStr HKLM "Software\RGPU" "NvEncBackedUp" "1"
    skip_nvenc_backup:
  no_nvenc_backup_needed:

  DetailPrint "Installing RGPU NVENC interpose as $SYSDIR\nvEncodeAPI64.dll..."
  CopyFiles /SILENT "$INSTDIR\lib\rgpu_nvenc_interpose.dll" "$SYSDIR\nvEncodeAPI64.dll"
  WriteRegStr HKLM "Software\RGPU" "NvencInterposed" "1"

  ${EnableX64FSRedirection}

  ; ---- NVDEC Interpose ----
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_nvdec_interpose.dll"

  ${DisableX64FSRedirection}

  IfFileExists "$SYSDIR\nvcuvid.dll" 0 no_nvdec_backup_needed
    IfFileExists "$SYSDIR\nvcuvid_real.dll" skip_nvdec_backup
      DetailPrint "Backing up original nvcuvid.dll..."
      CopyFiles /SILENT "$SYSDIR\nvcuvid.dll" "$SYSDIR\nvcuvid_real.dll"
      WriteRegStr HKLM "Software\RGPU" "NvDecBackedUp" "1"
    skip_nvdec_backup:
  no_nvdec_backup_needed:

  DetailPrint "Installing RGPU NVDEC interpose as $SYSDIR\nvcuvid.dll..."
  CopyFiles /SILENT "$INSTDIR\lib\rgpu_nvdec_interpose.dll" "$SYSDIR\nvcuvid.dll"
  WriteRegStr HKLM "Software\RGPU" "NvdecInterposed" "1"

  ${EnableX64FSRedirection}

  ; ---- NVML Interpose ----
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_nvml_interpose.dll"

  ${DisableX64FSRedirection}

  ; NVML: nvml.dll is typically locked by NVIDIA background services.
  ; Strategy: try direct rename+copy first; if file is locked, schedule
  ; replacement at reboot via MoveFileEx (runs before services start).

  IfFileExists "$SYSDIR\nvml.dll" 0 nvml_no_existing
    IfFileExists "$SYSDIR\nvml_real.dll" nvml_try_replace

    DetailPrint "Backing up original nvml.dll..."
    Rename "$SYSDIR\nvml.dll" "$SYSDIR\nvml_real.dll"
    IfErrors 0 nvml_try_replace

      DetailPrint "nvml.dll is locked - scheduling replacement for reboot..."
      CopyFiles /SILENT "$INSTDIR\lib\rgpu_nvml_interpose.dll" "$SYSDIR\nvml_rgpu_pending.dll"
      System::Call 'kernel32::MoveFileExW(t "$SYSDIR\nvml.dll", t "$SYSDIR\nvml_real.dll", i 5)'
      System::Call 'kernel32::MoveFileExW(t "$SYSDIR\nvml_rgpu_pending.dll", t "$SYSDIR\nvml.dll", i 5)'
      WriteRegStr HKLM "Software\RGPU" "NvmlBackedUp" "1"
      WriteRegStr HKLM "Software\RGPU" "NvmlInterposed" "1"
      SetRebootFlag true
      ${EnableX64FSRedirection}
      Goto nvml_done

  nvml_try_replace:
  Delete "$SYSDIR\nvml.dll"
  nvml_no_existing:
  DetailPrint "Installing RGPU NVML interpose as $SYSDIR\nvml.dll..."
  CopyFiles /SILENT "$INSTDIR\lib\rgpu_nvml_interpose.dll" "$SYSDIR\nvml.dll"
  WriteRegStr HKLM "Software\RGPU" "NvmlBackedUp" "1"
  WriteRegStr HKLM "Software\RGPU" "NvmlInterposed" "1"
  ${EnableX64FSRedirection}

  nvml_done:

  ; ---- NVAPI Interpose ----
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_nvapi_interpose.dll"

  ${DisableX64FSRedirection}

  IfFileExists "$SYSDIR\nvapi64.dll" 0 no_nvapi_backup_needed
    IfFileExists "$SYSDIR\nvapi64_real.dll" skip_nvapi_backup
      DetailPrint "Backing up original nvapi64.dll..."
      CopyFiles /SILENT "$SYSDIR\nvapi64.dll" "$SYSDIR\nvapi64_real.dll"
      WriteRegStr HKLM "Software\RGPU" "NvApiBackedUp" "1"
    skip_nvapi_backup:
  no_nvapi_backup_needed:

  DetailPrint "Installing RGPU NVAPI interpose as $SYSDIR\nvapi64.dll..."
  CopyFiles /SILENT "$INSTDIR\lib\rgpu_nvapi_interpose.dll" "$SYSDIR\nvapi64.dll"
  WriteRegStr HKLM "Software\RGPU" "NvapiInterposed" "1"

  ${EnableX64FSRedirection}

  ; ---- Vulkan ICD Driver ----
  SetOutPath "$INSTDIR\lib"
  File "staging\rgpu_vk_icd.dll"

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

  WriteRegDWORD HKLM "SOFTWARE\Khronos\Vulkan\Drivers" \
    "$INSTDIR\lib\rgpu_icd.json" 0

  ; ---- Client Daemon Service (auto-start with OS) ----
  nsExec::ExecToLog 'sc create "RGPU Client" binPath= "\"$INSTDIR\bin\rgpu.exe\" client --service --config \"$ProgramDataDir\RGPU\rgpu.toml\"" start= auto DisplayName= "RGPU Client Daemon"'
  nsExec::ExecToLog 'sc description "RGPU Client" "RGPU Client Daemon - connects to remote GPU servers and exposes them locally via IPC"'
  nsExec::ExecToLog 'sc start "RGPU Client"'

  ; Record installation type
  WriteRegStr HKLM "Software\RGPU" "InstallType" "Client"
SectionEnd

;---------------------------------
; Installation Type: Server
; Installs RGPU Server as a Windows Service (auto-start with OS).
; For machines with GPUs to share over the network.
;---------------------------------
Section /o "Server Installation" SEC_SERVER
  ; ---- Server Windows Service (auto-start with OS) ----
  nsExec::ExecToLog 'sc create "RGPU Server" binPath= "\"$INSTDIR\bin\rgpu.exe\" server --service --config \"$ProgramDataDir\RGPU\rgpu.toml\"" start= auto DisplayName= "RGPU Remote GPU Server"'
  nsExec::ExecToLog 'sc description "RGPU Server" "RGPU Remote GPU Sharing Server - exposes local GPUs over the network"'
  nsExec::ExecToLog 'sc start "RGPU Server"'

  ; Record installation type
  WriteRegStr HKLM "Software\RGPU" "InstallType" "Server"
SectionEnd

;---------------------------------
; Enforce mutual exclusivity: Client and Server cannot both be selected
;---------------------------------
Function .onSelChange
  ; Get which section changed by checking current state
  SectionGetFlags ${SEC_CLIENT} $0
  SectionGetFlags ${SEC_SERVER} $1
  IntOp $0 $0 & 1
  IntOp $1 $1 & 1

  ; If both are selected, warn and deselect the other
  IntCmp $0 0 check_none
  IntCmp $1 0 no_conflict
    ; Both selected - deselect server (client was most recently toggled)
    MessageBox MB_OK|MB_ICONEXCLAMATION "Client and Server cannot be installed on the same machine.$\r$\nInstalling interpose libraries on a server would replace NVIDIA DLLs and break direct GPU access.$\r$\n$\r$\nPlease select only one."
    SectionGetFlags ${SEC_SERVER} $0
    IntOp $0 $0 & 0xFFFFFFFE
    SectionSetFlags ${SEC_SERVER} $0
    Goto no_conflict

  check_none:
  ; If neither is selected, that's fine - user can choose later
  no_conflict:
FunctionEnd

;---------------------------------
; Section Descriptions
;---------------------------------
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CORE} \
    "The RGPU command-line tool and GUI. Required."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_CLIENT} \
    "Install RGPU as a client machine. Installs all GPU interpose libraries (CUDA, NVENC, NVDEC, NVML, NVAPI), Vulkan ICD driver, and the RGPU Client Daemon service (auto-starts with Windows). Use this on machines that need to access remote GPUs."
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_SERVER} \
    "Install RGPU as a server machine. Installs the RGPU Server service (auto-starts with Windows). Use this on machines with GPUs that you want to share over the network."
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
  ; NOTE: NSIS is 32-bit, so we must disable WOW64 redirection to access real System32
  ReadRegStr $0 HKLM "Software\RGPU" "CudaInterposed"
  StrCmp $0 "1" 0 skip_cuda_restore
    ${DisableX64FSRedirection}
    ; Check if backup exists
    IfFileExists "$SYSDIR\nvcuda_real.dll" 0 remove_our_nvcuda
      ; Restore the original NVIDIA driver
      DetailPrint "Restoring original nvcuda.dll from backup..."
      Delete "$SYSDIR\nvcuda.dll"
      CopyFiles /SILENT "$SYSDIR\nvcuda_real.dll" "$SYSDIR\nvcuda.dll"
      Delete "$SYSDIR\nvcuda_real.dll"
      ${EnableX64FSRedirection}
      Goto skip_cuda_restore
    remove_our_nvcuda:
      ; No backup existed (no NVIDIA driver was installed), just remove our DLL
      Delete "$SYSDIR\nvcuda.dll"
      ${EnableX64FSRedirection}
  skip_cuda_restore:

  ; Restore original nvEncodeAPI64.dll (NVENC)
  ReadRegStr $0 HKLM "Software\RGPU" "NvencInterposed"
  StrCmp $0 "1" 0 skip_nvenc_restore
    ${DisableX64FSRedirection}
    IfFileExists "$SYSDIR\nvEncodeAPI64_real.dll" 0 remove_our_nvenc
      DetailPrint "Restoring original nvEncodeAPI64.dll from backup..."
      Delete "$SYSDIR\nvEncodeAPI64.dll"
      CopyFiles /SILENT "$SYSDIR\nvEncodeAPI64_real.dll" "$SYSDIR\nvEncodeAPI64.dll"
      Delete "$SYSDIR\nvEncodeAPI64_real.dll"
      ${EnableX64FSRedirection}
      Goto skip_nvenc_restore
    remove_our_nvenc:
      Delete "$SYSDIR\nvEncodeAPI64.dll"
      ${EnableX64FSRedirection}
  skip_nvenc_restore:

  ; Restore original nvcuvid.dll (NVDEC)
  ReadRegStr $0 HKLM "Software\RGPU" "NvdecInterposed"
  StrCmp $0 "1" 0 skip_nvdec_restore
    ${DisableX64FSRedirection}
    IfFileExists "$SYSDIR\nvcuvid_real.dll" 0 remove_our_nvdec
      DetailPrint "Restoring original nvcuvid.dll from backup..."
      Delete "$SYSDIR\nvcuvid.dll"
      CopyFiles /SILENT "$SYSDIR\nvcuvid_real.dll" "$SYSDIR\nvcuvid.dll"
      Delete "$SYSDIR\nvcuvid_real.dll"
      ${EnableX64FSRedirection}
      Goto skip_nvdec_restore
    remove_our_nvdec:
      Delete "$SYSDIR\nvcuvid.dll"
      ${EnableX64FSRedirection}
  skip_nvdec_restore:

  ; Restore original nvml.dll (NVML)
  ; Our nvml.dll may be locked by NVIDIA services; use MoveFileEx reboot fallback.
  ReadRegStr $0 HKLM "Software\RGPU" "NvmlInterposed"
  StrCmp $0 "1" 0 skip_nvml_restore
    ${DisableX64FSRedirection}
    IfFileExists "$SYSDIR\nvml_real.dll" 0 un_remove_our_nvml
      ; Try direct restore: delete ours, copy backup back, delete backup
      DetailPrint "Restoring original nvml.dll from backup..."
      Delete "$SYSDIR\nvml.dll"
      IfErrors 0 un_nvml_direct_restore
        ; Delete failed (file locked) - schedule for reboot
        DetailPrint "nvml.dll is locked - scheduling restore for reboot..."
        System::Call 'kernel32::MoveFileExW(t "$SYSDIR\nvml.dll", t 0, i 4)'
        System::Call 'kernel32::MoveFileExW(t "$SYSDIR\nvml_real.dll", t "$SYSDIR\nvml.dll", i 5)'
        SetRebootFlag true
        ${EnableX64FSRedirection}
        Goto skip_nvml_restore
      un_nvml_direct_restore:
      CopyFiles /SILENT "$SYSDIR\nvml_real.dll" "$SYSDIR\nvml.dll"
      Delete "$SYSDIR\nvml_real.dll"
      ${EnableX64FSRedirection}
      Goto skip_nvml_restore
    un_remove_our_nvml:
      ; No backup - just remove our DLL
      Delete "$SYSDIR\nvml.dll"
      IfErrors 0 +3
        ; Locked - schedule deletion for reboot
        System::Call 'kernel32::MoveFileExW(t "$SYSDIR\nvml.dll", t 0, i 4)'
        SetRebootFlag true
      ${EnableX64FSRedirection}
  skip_nvml_restore:

  ; Clean up any pending NVML staging file (from a reboot-deferred install)
  ${DisableX64FSRedirection}
  Delete "$SYSDIR\nvml_rgpu_pending.dll"
  ${EnableX64FSRedirection}

  ; Restore original nvapi64.dll (NVAPI)
  ReadRegStr $0 HKLM "Software\RGPU" "NvapiInterposed"
  StrCmp $0 "1" 0 skip_nvapi_restore
    ${DisableX64FSRedirection}
    IfFileExists "$SYSDIR\nvapi64_real.dll" 0 remove_our_nvapi
      DetailPrint "Restoring original nvapi64.dll from backup..."
      Delete "$SYSDIR\nvapi64.dll"
      CopyFiles /SILENT "$SYSDIR\nvapi64_real.dll" "$SYSDIR\nvapi64.dll"
      Delete "$SYSDIR\nvapi64_real.dll"
      ${EnableX64FSRedirection}
      Goto skip_nvapi_restore
    remove_our_nvapi:
      Delete "$SYSDIR\nvapi64.dll"
      ${EnableX64FSRedirection}
  skip_nvapi_restore:

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
  Delete "$INSTDIR\lib\rgpu_nvenc_interpose.dll"
  Delete "$INSTDIR\lib\rgpu_nvdec_interpose.dll"
  Delete "$INSTDIR\lib\rgpu_nvml_interpose.dll"
  Delete "$INSTDIR\lib\rgpu_nvapi_interpose.dll"
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
