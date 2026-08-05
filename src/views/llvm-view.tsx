import {h, JSX} from 'preact'
import {useState} from 'preact/hooks'
import {StyleSheet, css} from 'aphrodite'
import {ActiveProfileState} from '../app-state/active-profile-state'
import {profileGroupAtom} from '../app-state'
import {useTheme, withTheme} from './themes/theme'
import {FontFamily, FontSize} from './style'
import {GearIcon, RocketIcon, BoxIcon, ZapIcon, FolderIcon} from './icons'

interface LLVMViewProps {
  activeProfileState: ActiveProfileState
}

export function LLVMView({activeProfileState}: LLVMViewProps): JSX.Element {
  const style = getStyle(useTheme())
  const profile = activeProfileState.profile

  const [sidebarQuery, setSidebarQuery] = useState('')
  const [codeQuery, setCodeQuery] = useState('')

  const rawProfile = (profile as any)?.rawProfile || (window as any).gRawProfile || (profileGroupAtom.get()?.profiles?.[0] as any)?.profile?.rawProfile
  const llvmMap = rawProfile?.shared?.llvm_map || {}

  const frameList: Array<{key: string | number; name: string; file?: string; line?: number}> = []
  const addedNames = new Set<string>()

  // 1. Add all JIT compiled functions recorded in llvm_map
  Object.keys(llvmMap).forEach(key => {
    const item = llvmMap[key]
    frameList.push({
      key: key,
      name: key,
      file: item.file || 'Numba JIT Compiled Function',
      line: item.line,
    })
    addedNames.add(key.toLowerCase())
  })

  // 2. Add profile frames that match llvm_map or are user code (excluding stdlib / loader machinery)
  const ignoredStdlib = ['importlib', '_bootstrap', 'threading.py', 'runpy.py', 'contextlib.py', '<module>', '<lambda>', 'njit']
  profile.forEachFrame(f => {
    if (!f.name) return
    const lowerName = f.name.toLowerCase()
    if (ignoredStdlib.some(ig => lowerName.includes(ig))) return

    const clean = f.name.replace(/\s*\(.*\)/, '').trim()
    const short = clean.split('.').pop()!

    if (!addedNames.has(clean.toLowerCase()) && !addedNames.has(short.toLowerCase())) {
      frameList.push({
        key: f.key,
        name: f.name,
        file: f.file,
        line: f.line,
      })
      addedNames.add(clean.toLowerCase())
    }
  })

  const filteredFrames = frameList.filter(f =>
    f.name.toLowerCase().includes(sidebarQuery.toLowerCase()) ||
    (f.file && f.file.toLowerCase().includes(sidebarQuery.toLowerCase()))
  )

  const getLLVMMapItem = (funcName: string) => {
    if (!funcName) return null
    let cleanName = funcName.replace(/\s*\(.*\)/, '').trim()
    if (cleanName.includes('.')) {
      const parts = cleanName.split('.')
      cleanName = parts[parts.length - 1]
    }
    const rawProfile = (profile as any)?.rawProfile || (window as any).gRawProfile || (profileGroupAtom.get()?.profiles?.[0] as any)?.profile?.rawProfile
    const map = rawProfile?.shared?.llvm_map
    if (map) {
      if (map[funcName]) return map[funcName]
      if (map[cleanName]) return map[cleanName]
      for (const k of Object.keys(map)) {
        const cleanKey = k.includes('.') ? k.split('.').pop()! : k
        if (cleanKey.toLowerCase() === cleanName.toLowerCase()) return map[k]
      }
      for (const k of Object.keys(map)) {
        const cleanKey = k.includes('.') ? k.split('.').pop()! : k
        if (cleanKey.length >= 5 && (cleanName.toLowerCase() === cleanKey.toLowerCase() || cleanName.toLowerCase().endsWith('_' + cleanKey.toLowerCase()))) {
          return map[k]
        }
      }
    }
    return null
  }

  const firstWithIR = filteredFrames.find(f => getLLVMMapItem(f.name) != null) || filteredFrames[0] || null
  const [selectedFrame, setSelectedFrame] = useState(firstWithIR)

  const getSimdStatus = (name: string) => {
    const item = getLLVMMapItem(name)
    if (item) {
      return {
        enabled: item.simd_vectorized,
        text: item.simd_vectorized
          ? `✨ SIMD Vectorization ENABLED: ${item.simd_instructions} SIMD instructions detected`
          : '⚠️ SIMD Vectorization NOT DETECTED: Consider enabling @njit(fastmath=True)',
        instructions: item.total_instructions,
        simdOps: item.simd_instructions,
        memAlloc: item.memory_allocations || 0,
      }
    }
    return {
      enabled: false,
      text: '⚠️ LLVM IR not recorded in shared.llvm_map',
      instructions: 0,
      simdOps: 0,
      memAlloc: 0,
    }
  }

  const getFunctionLLVMIR = (name: string) => {
    const item = getLLVMMapItem(name)
    if (item && item.llvm_ir) {
      return item.llvm_ir
    }

    const cleanName = name.replace(/\s*\(.*\)/, '').trim()
    return `; ModuleID = 'numba.compiled.${cleanName}'
; Function: ${cleanName}
; Status: Real LLVM IR not recorded in shared.llvm_map.
;
; Possible Reasons:
; 1. Function executed via C/Cython extension or CPython interpreter without Numba JIT.
; 2. Function loaded from pre-compiled Numba disk cache (.numba_cache).
; 3. Function is a decorator/wrapper (@njit) or standard library function.
;
; Remedy: Ensure Profila runs with live in-memory JIT compilation (--disable-cache):
;   uv run python -m profila viewer --disable-cache -- <your_script.py>`
  }

  const rawIrText = selectedFrame ? getFunctionLLVMIR(selectedFrame.name) : ''
  const irLines = rawIrText.split('\n')
  const matchingLineCount = codeQuery.trim()
    ? irLines.filter(l => l.toLowerCase().includes(codeQuery.toLowerCase())).length
    : 0

  return (
    <div className={css(style.container)}>
      <div className={css(style.sidebar)}>
        <div className={css(style.sidebarTitle)}>⚡ Numba Functions ({filteredFrames.length})</div>
        <div className={css(style.searchContainer)}>
          <input
            type="text"
            placeholder="🔍 Filter functions..."
            value={sidebarQuery}
            onInput={(e: any) => setSidebarQuery(e.target.value)}
            className={css(style.searchInput)}
          />
        </div>
        {filteredFrames.map(f => {
          const status = getSimdStatus(f.name)
          const isSelected = selectedFrame?.name === f.name
          return (
            <div
              key={f.key}
              className={css(style.frameItem, isSelected && style.frameItemSelected)}
              onClick={() => setSelectedFrame(f)}
            >
              <div className={css(style.frameName)}>{f.name}</div>
              <div className={css(style.frameSub)} style={{display: 'flex', alignItems: 'center', gap: '4px'}}>
                {status.enabled ? <><RocketIcon size={12} color="#2ecc71" /> SIMD Vectorized</> : <><GearIcon size={12} color="#e67e22" /> Scalar Execution</>}
              </div>
            </div>
          )
        })}
      </div>

      <div className={css(style.mainContent)}>
        {selectedFrame ? (
          <div>
            <div className={css(style.header)}>
              <h2 style={{display: 'flex', alignItems: 'center', gap: '8px'}}><GearIcon size={20} color="#e67e22" /> Function: {selectedFrame.name}</h2>
              <div className={css(style.filePath)} style={{display: 'flex', alignItems: 'center', gap: '4px'}}>
                <FolderIcon size={12} color="#888" /> {selectedFrame.file}:{selectedFrame.line}
              </div>
            </div>

            {(() => {
              const status = getSimdStatus(selectedFrame.name)
              return (
                <div className={css(style.statusBox, status.enabled ? style.statusEnabled : style.statusDisabled)}>
                  <div className={css(style.statusText)}>{status.text}</div>
                  <div className={css(style.metricsRow)}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '4px'}}><GearIcon size={14} /> Total LLVM Instructions: <b>{status.instructions}</b></div>
                    <div style={{display: 'flex', alignItems: 'center', gap: '4px'}}><RocketIcon size={14} /> SIMD Operations: <b>{status.simdOps}</b></div>
                    <div style={{display: 'flex', alignItems: 'center', gap: '4px'}}><BoxIcon size={14} /> Memory Allocations: <b>{status.memAlloc}</b></div>
                  </div>
                </div>
              )
            })()}

            <div className={css(style.codeHeaderRow)}>
              <div className={css(style.codeTitle)}>Generated LLVM IR Assembly:</div>
              <div className={css(style.codeSearchWrapper)}>
                <input
                  type="text"
                  placeholder="🔍 Search LLVM IR instructions..."
                  value={codeQuery}
                  onInput={(e: any) => setCodeQuery(e.target.value)}
                  className={css(style.codeSearchInput)}
                />
                {codeQuery.trim() && (
                  <span className={css(style.matchBadge)}>{matchingLineCount} matches</span>
                )}
              </div>
            </div>

            <div className={css(style.codeBlock)}>
              {irLines.map((line, idx) => {
                const isMatch = codeQuery.trim() && line.toLowerCase().includes(codeQuery.toLowerCase())
                return (
                  <div key={idx} className={css(isMatch && style.highlightedLine)}>
                    {line}
                  </div>
                )
              })}
            </div>
          </div>
        ) : (
          <div className={css(style.placeholder)}>Select a Numba function from the left sidebar to inspect LLVM IR.</div>
        )}
      </div>
    </div>
  )
}

const getStyle = withTheme(theme =>
  StyleSheet.create({
    container: {
      display: 'flex',
      flex: 1,
      height: '100%',
      background: theme.bgPrimaryColor,
      color: theme.fgPrimaryColor,
      fontFamily: FontFamily.MONOSPACE,
    },
    sidebar: {
      width: 320,
      borderRight: `1px solid ${theme.altBgSecondaryColor}`,
      background: theme.bgSecondaryColor,
      overflowY: 'auto',
    },
    sidebarTitle: {
      padding: 12,
      fontWeight: 'bold',
      fontSize: FontSize.TITLE,
      borderBottom: `1px solid ${theme.altBgSecondaryColor}`,
      background: theme.altBgPrimaryColor,
    },
    searchContainer: {
      padding: 8,
      borderBottom: `1px solid ${theme.altBgSecondaryColor}`,
    },
    searchInput: {
      width: '100%',
      padding: '6px 10px',
      borderRadius: 4,
      border: '1px solid #333',
      background: '#121212',
      color: '#fff',
      fontSize: 12,
      fontFamily: FontFamily.MONOSPACE,
      outline: 'none',
      ':focus': {
        borderColor: theme.selectionPrimaryColor,
      },
    },
    frameItem: {
      padding: '10px 12px',
      cursor: 'pointer',
      borderBottom: `1px solid ${theme.altBgSecondaryColor}`,
      ':hover': {
        background: theme.selectionSecondaryColor,
      },
    },
    frameItemSelected: {
      background: theme.selectionPrimaryColor,
      color: '#fff',
      ':hover': {
        background: theme.selectionPrimaryColor,
      },
    },
    frameName: {
      fontWeight: 'bold',
      fontSize: FontSize.LABEL,
    },
    frameSub: {
      fontSize: 11,
      opacity: 0.8,
      marginTop: 4,
    },
    mainContent: {
      flex: 1,
      padding: 20,
      overflowY: 'auto',
    },
    header: {
      marginBottom: 16,
    },
    filePath: {
      color: theme.fgSecondaryColor,
      fontSize: FontSize.LABEL,
      marginTop: 4,
    },
    statusBox: {
      padding: 16,
      borderRadius: 6,
      marginBottom: 20,
    },
    statusEnabled: {
      background: 'rgba(46, 204, 113, 0.15)',
      border: '1px solid #2ecc71',
      color: '#2ecc71',
    },
    statusDisabled: {
      background: 'rgba(230, 126, 34, 0.15)',
      border: '1px solid #e67e22',
      color: '#e67e22',
    },
    statusText: {
      fontWeight: 'bold',
      fontSize: FontSize.TITLE,
      marginBottom: 8,
    },
    metricsRow: {
      display: 'flex',
      gap: 20,
      fontSize: FontSize.LABEL,
      color: theme.fgPrimaryColor,
    },
    codeHeaderRow: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 8,
    },
    codeTitle: {
      fontWeight: 'bold',
    },
    codeSearchWrapper: {
      display: 'flex',
      alignItems: 'center',
      gap: 10,
    },
    codeSearchInput: {
      width: 260,
      padding: '5px 10px',
      borderRadius: 4,
      border: '1px solid #333',
      background: '#121212',
      color: '#fff',
      fontSize: 12,
      fontFamily: FontFamily.MONOSPACE,
      outline: 'none',
      ':focus': {
        borderColor: theme.selectionPrimaryColor,
      },
    },
    matchBadge: {
      fontSize: 11,
      color: '#2ecc71',
      fontWeight: 'bold',
    },
    codeBlock: {
      background: '#121212',
      padding: 16,
      borderRadius: 6,
      border: '1px solid #333',
      color: '#38bdf8',
      fontSize: 12,
      lineHeight: '20px',
      overflowX: 'auto',
      whiteSpace: 'pre-wrap',
    },
    highlightedLine: {
      background: 'rgba(230, 126, 34, 0.3)',
      color: '#fff',
      fontWeight: 'bold',
      borderRadius: 2,
    },
    placeholder: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      color: theme.fgSecondaryColor,
    },
  }),
)
