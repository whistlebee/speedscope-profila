import {h, JSX} from 'preact'
import {useState} from 'preact/hooks'
import {StyleSheet, css} from 'aphrodite'
import {Frame} from '../lib/profile'
import {profileGroupAtom} from '../app-state'
import {useTheme, withTheme} from './themes/theme'
import {FontFamily, FontSize} from './style'
import {GearIcon, RocketIcon, FolderIcon, BoxIcon} from './icons'

interface LLVMSideDrawerProps {
  selectedFrame: Frame
  onClose: () => void
}

export function LLVMSideDrawer({selectedFrame, onClose}: LLVMSideDrawerProps): JSX.Element {
  const style = getStyle(useTheme())
  const [codeQuery, setCodeQuery] = useState('')

  const name = selectedFrame.name || 'Unknown Function'
  const file = selectedFrame.file || ''
  const line = selectedFrame.line

  const getLLVMMapItem = (funcName: string) => {
    if (!funcName) return null
    let cleanName = funcName.replace(/\s*\(.*\)/, '').trim()
    if (cleanName.includes('.')) {
      const parts = cleanName.split('.')
      cleanName = parts[parts.length - 1]
    }
    const pg = typeof profileGroupAtom !== 'undefined' ? profileGroupAtom.get() : null
    const activeProfile = (pg?.profiles?.[0] as any)?.profile
    const rawProfile = (window as any).gRawProfile || activeProfile?.rawProfile || (selectedFrame as any)?.rawProfile
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

  const getSimdStatus = (funcName: string) => {
    const item = getLLVMMapItem(funcName)
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

  const getFunctionLLVMIR = (funcName: string) => {
    const item = getLLVMMapItem(funcName)
    if (item && item.llvm_ir) {
      return item.llvm_ir
    }

    const cleanName = funcName.replace(/\s*\(.*\)/, '').trim()
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

  const status = getSimdStatus(name)
  const rawIrText = getFunctionLLVMIR(name)
  const irLines = rawIrText.split('\n')
  const matchingLineCount = codeQuery.trim()
    ? irLines.filter(l => l.toLowerCase().includes(codeQuery.toLowerCase())).length
    : 0

  const handleClose = (e: any) => {
    if (e) {
      e.preventDefault()
      e.stopPropagation()
    }
    onClose()
  }

  return (
    <div
      className={css(style.drawerContainer)}
      onClick={(e: any) => e.stopPropagation()}
      onMouseDown={(e: any) => e.stopPropagation()}
    >
      <div className={css(style.drawerHeader)}>
        <div className={css(style.drawerTitle)} style={{display: 'flex', alignItems: 'center', gap: '6px'}}>
          <GearIcon size={16} color="#e67e22" /> LLVM IR & SIMD Inspector
        </div>
        <button
          className={css(style.closeButton)}
          onClick={handleClose}
          onMouseDown={handleClose}
          title="Close Side Panel"
        >
          ✖
        </button>
      </div>

      <div className={css(style.funcName)}>{name}</div>
      {file && (
        <div className={css(style.filePath)} style={{display: 'flex', alignItems: 'center', gap: '4px'}}>
          <FolderIcon size={12} color="#888" /> {file}:{line}
        </div>
      )}

      <div className={css(style.statusBox, status.enabled ? style.statusEnabled : style.statusDisabled)}>
        <div className={css(style.statusText)}>{status.text}</div>
        <div className={css(style.metricsRow)}>
          <div style={{display: 'flex', alignItems: 'center', gap: '4px'}}><GearIcon size={12} /> Instructions: <b>{status.instructions}</b></div>
          <div style={{display: 'flex', alignItems: 'center', gap: '4px'}}><RocketIcon size={12} /> SIMD Ops: <b>{status.simdOps}</b></div>
        </div>
      </div>

      <div className={css(style.codeHeaderRow)}>
        <div className={css(style.codeTitle)}>LLVM IR Assembly:</div>
        <input
          type="text"
          placeholder="🔍 Search instructions..."
          value={codeQuery}
          onInput={(e: any) => setCodeQuery(e.target.value)}
          className={css(style.codeSearchInput)}
        />
      </div>

      {codeQuery.trim() ? (
        <div className={css(style.matchBadge)}>{matchingLineCount} matching lines</div>
      ) : null}

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
  )
}

const getStyle = withTheme(theme =>
  StyleSheet.create({
    drawerContainer: {
      position: 'absolute',
      top: 0,
      right: 0,
      width: 440,
      height: '100%',
      background: '#161616',
      borderLeft: '2px solid #e67e22',
      boxShadow: '-4px 0 20px rgba(0, 0, 0, 0.6)',
      zIndex: 1000,
      padding: 16,
      display: 'flex',
      flexDirection: 'column',
      color: '#fff',
      fontFamily: FontFamily.MONOSPACE,
      overflowY: 'auto',
    },
    drawerHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      paddingBottom: 10,
      borderBottom: '1px solid #333',
      marginBottom: 12,
    },
    drawerTitle: {
      fontWeight: 'bold',
      fontSize: FontSize.TITLE,
      color: '#e67e22',
    },
    closeButton: {
      background: 'transparent',
      border: 'none',
      color: '#aaa',
      fontSize: 16,
      cursor: 'pointer',
      ':hover': {
        color: '#fff',
      },
    },
    funcName: {
      fontWeight: 'bold',
      fontSize: 14,
      color: '#38bdf8',
      marginBottom: 4,
      wordBreak: 'break-all',
    },
    filePath: {
      fontSize: 11,
      color: '#888',
      marginBottom: 12,
    },
    statusBox: {
      padding: 12,
      borderRadius: 6,
      marginBottom: 14,
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
      fontSize: 12,
      marginBottom: 6,
    },
    metricsRow: {
      display: 'flex',
      gap: 16,
      fontSize: 11,
      color: '#ddd',
    },
    codeHeaderRow: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 6,
    },
    codeTitle: {
      fontWeight: 'bold',
      fontSize: 12,
    },
    codeSearchInput: {
      width: 180,
      padding: '4px 8px',
      borderRadius: 4,
      border: '1px solid #333',
      background: '#0d0d0d',
      color: '#fff',
      fontSize: 11,
      fontFamily: FontFamily.MONOSPACE,
      outline: 'none',
    },
    matchBadge: {
      fontSize: 10,
      color: '#2ecc71',
      marginBottom: 6,
    },
    codeBlock: {
      background: '#0a0a0a',
      padding: 12,
      borderRadius: 6,
      border: '1px solid #262626',
      color: '#38bdf8',
      fontSize: 11,
      lineHeight: '18px',
      flex: 1,
      overflowX: 'auto',
      whiteSpace: 'pre-wrap',
    },
    highlightedLine: {
      background: 'rgba(230, 126, 34, 0.35)',
      color: '#fff',
      fontWeight: 'bold',
    },
  }),
)
