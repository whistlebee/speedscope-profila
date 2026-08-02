import {Color} from '../../lib/color'
import {triangle} from '../../lib/utils'
import {Theme} from './theme'

// These colors are intentionally not exported from this file, because these
// colors are theme specific, and we want all color values to come from the
// active theme.
enum Colors {
  LIGHTER_GRAY = '#E0E0E0',
  LIGHT_GRAY = '#CCCCCC',
  GRAY = '#858585',
  DARK_GRAY = '#252526',
  DARKER_GRAY = '#1E1E1E',
  OFF_BLACK = '#181818',
  BLACK = '#141414',
  PROFILA_ORANGE = '#E67E22',
  PROFILA_DARK_ORANGE = '#D35400',
  BLUE = '#007ACC',
  PALE_BLUE = '#004E75',
  GREEN = '#2ECC71',
  LIGHT_BROWN = '#D6AE24',
  BROWN = '#A66F1C',
}

const C_0 = 0.25
const C_d = 0.15
const L_0 = 0.35
const L_d = 0.15

const colorForBucket = (t: number) => {
  const x = triangle(30.0 * t)
  const H = 360.0 * (0.95 * t)
  const C = C_0 + C_d * x
  const L = L_0 - L_d * x
  return Color.fromLumaChromaHue(L, C, H)
}
const colorForBucketGLSL = `
  vec3 colorForBucket(float t) {
    float x = triangle(30.0 * t);
    float H = 360.0 * (0.95 * t);
    float C = ${C_0.toFixed(2)} + ${C_d.toFixed(2)} * x;
    float L = ${L_0.toFixed(2)} - ${L_d.toFixed(2)} * x;
    return hcl2rgb(H, C, L);
  }
`

export const darkTheme: Theme = {
  fgPrimaryColor: Colors.LIGHTER_GRAY,
  fgSecondaryColor: Colors.GRAY,

  bgPrimaryColor: Colors.OFF_BLACK,
  bgSecondaryColor: Colors.DARKER_GRAY,

  altFgPrimaryColor: Colors.LIGHTER_GRAY,
  altFgSecondaryColor: Colors.GRAY,

  altBgPrimaryColor: Colors.BLACK,
  altBgSecondaryColor: Colors.DARK_GRAY,

  selectionPrimaryColor: Colors.PROFILA_ORANGE,
  selectionSecondaryColor: Colors.PROFILA_DARK_ORANGE,

  weightColor: Colors.GREEN,

  searchMatchTextColor: Colors.DARKER_GRAY,
  searchMatchPrimaryColor: Colors.PROFILA_ORANGE,
  searchMatchSecondaryColor: Colors.PROFILA_DARK_ORANGE,

  colorForBucket,
  colorForBucketGLSL,
}
