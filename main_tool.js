// ==================== ImageProcessor 类（原样保留） ====================
class ImageProcessor {
    constructor(imageSrc) {
        this.src = imageSrc;
        this.imageData = null;
        this.canvas = null;
        this.ctx = null;
    }

    load() {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                this.imageData = ctx.getImageData(0, 0, img.width, img.height);
                this.canvas = canvas;
                this.ctx = ctx;
                resolve(this);
            };
            img.onerror = reject;
            img.src = this.src;
        });
    }

    getPixel(x, y) {
        if (!this.imageData) throw new Error('图片尚未加载');
        const idx = (y * this.imageData.width + x) * 4;
        const d = this.imageData.data;
        return { r: d[idx], g: d[idx + 1], b: d[idx + 2], a: d[idx + 3] };
    }

    setPixel(x, y, r, g, b, a = 255) {
        if (!this.imageData) throw new Error('图片尚未加载');
        const idx = (y * this.imageData.width + x) * 4;
        const d = this.imageData.data;
        d[idx] = r; d[idx + 1] = g; d[idx + 2] = b; d[idx + 3] = a;
    }

    forEachPixel(callback) {
        if (!this.imageData) throw new Error('图片尚未加载');
        const d = this.imageData.data;
        const { width, height } = this.imageData;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const pixel = { r: d[idx], g: d[idx + 1], b: d[idx + 2], a: d[idx + 3] };
                const newPixel = callback(pixel, x, y);
                if (newPixel) {
                    d[idx] = newPixel.r ?? pixel.r;
                    d[idx + 1] = newPixel.g ?? pixel.g;
                    d[idx + 2] = newPixel.b ?? pixel.b;
                    d[idx + 3] = newPixel.a ?? pixel.a;
                }
            }
        }
    }

    toDataURL(type = 'image/png', quality = 0.92) {
        if (!this.ctx) throw new Error('图片尚未加载');
        this.ctx.putImageData(this.imageData, 0, 0);
        return this.canvas.toDataURL(type, quality);
    }

    get width() { return this.imageData?.width || 0; }
    get height() { return this.imageData?.height || 0; }
}

// ==================== 文字 ↔ 比特（标准 UTF-8） ====================
function textToBits(text) {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    const bits = [];
    for (const byte of bytes) {
        for (let i = 7; i >= 0; i--) {
            bits.push((byte >> i) & 1);
        }
    }
    return bits;
}

function bitsToText(bits) {
    const bytes = [];
    for (let i = 0; i < bits.length; i += 8) {
        let byte = 0;
        for (let j = 0; j < 8; j++) {
            byte = (byte << 1) | (bits[i + j] || 0);
        }
        bytes.push(byte);
    }
    const decoder = new TextDecoder();
    return decoder.decode(new Uint8Array(bytes));
}

// ==================== DCT / IDCT（8x8，修正缩放系数） ====================
function dct1d(input) {
    const N = 8;
    const out = new Float64Array(N);
    const scale = Math.sqrt(2 / N);   // ← 修正：从 (2/N) 改为 sqrt(2/N)
    for (let k = 0; k < N; k++) {
        let sum = 0;
        for (let n = 0; n < N; n++) {
            sum += input[n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
        }
        const alpha = k === 0 ? 1 / Math.sqrt(2) : 1;
        out[k] = scale * alpha * sum;
    }
    return out;
}

function idct1d(input) {
    const N = 8;
    const out = new Float64Array(N);
    const scale = Math.sqrt(2 / N);   // ← 修正：从 (2/N) 改为 sqrt(2/N)
    for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < N; k++) {
            const alpha = k === 0 ? 1 / Math.sqrt(2) : 1;
            sum += alpha * input[k] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
        }
        out[n] = scale * sum;
    }
    return out;
}

function dct2d(matrix) {
    const N = 8;
    const rows = matrix.map(row => dct1d(row));
    const result = Array.from({ length: N }, (_, i) => {
        const col = rows.map(row => row[i]);
        return dct1d(col);
    });
    // 转置
    return Array.from({ length: N }, (_, i) =>
        Array.from({ length: N }, (_, j) => result[j][i])
    );
}

function idct2d(matrix) {
    const N = 8;
    const cols = Array.from({ length: N }, (_, i) => {
        const col = matrix.map(row => row[i]);
        return idct1d(col);
    });
    const result = Array.from({ length: N }, (_, i) => {
        const row = cols.map(col => col[i]);
        return idct1d(row);
    });
    return result;
}

// ==================== 灰度计算（标准 luminance） ====================
function rgbToGray(r, g, b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

// ==================== 嵌入 / 提取单个比特（基于 8×8 灰度块） ====================
function embedBitInGrayBlock(block, bit) {
    const dct = dct2d(block);
    const u = 4, v = 3;
    let coeff = dct[u][v];
    const parity = Math.round(coeff) % 2;
    if (parity !== bit) {
        coeff = Math.round(coeff) + (bit === 1 ? 1 : -1);
        dct[u][v] = coeff;
    }
    const idct = idct2d(dct);
    return idct.map(row => row.map(val => Math.round(Math.min(255, Math.max(0, val)))));
}

function extractBitFromGrayBlock(block) {
    const dct = dct2d(block);
    return Math.round(dct[4][3]) % 2;
}

// ==================== 嵌入文字水印（保持色度比例，无色偏） ====================
function embedTextWatermark(processor, text) {
    const data = processor.imageData.data;
    const w = processor.width, h = processor.height;
    const bits = textToBits(text);
    const bitLen = bits.length;
    let bitIdx = 0;
    const log = [];  // 记录每个块的信息

    for (let y = 0; y < h - 7; y += 8) {
        for (let x = 0; x < w - 7; x += 8) {
            // 提取灰度块
            const block = [];
            for (let dy = 0; dy < 8; dy++) {
                const row = [];
                for (let dx = 0; dx < 8; dx++) {
                    const idx = ((y + dy) * w + (x + dx)) * 4;
                    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                    row.push(rgbToGray(r, g, b));
                }
                block.push(row);
            }

            const bit = bits[bitIdx % bitLen];
            // 计算嵌入前后的 DCT 系数
            const dctBefore = dct2d(block);
            const coeffBefore = dctBefore[4][3];
            const newGrayBlock = embedBitInGrayBlock(block, bit);
            const dctAfter = dct2d(newGrayBlock);
            const coeffAfter = dctAfter[4][3];
bitIdx++;
            // 3. 写回 RGB（保持色度比例）
            for (let dy = 0; dy < 8; dy++) {
                for (let dx = 0; dx < 8; dx++) {
                    const idx = ((y + dy) * w + (x + dx)) * 4;
                    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                    const oldGray = block[dy][dx];
                    const newGray = newGrayBlock[dy][dx];
                    let ratioR, ratioG, ratioB;
                    if (oldGray > 0.5) {
                        ratioR = r / oldGray;
                        ratioG = g / oldGray;
                        ratioB = b / oldGray;
                    } else {
                        ratioR = 0.299;
                        ratioG = 0.587;
                        ratioB = 0.114;
                    }
                    data[idx] = Math.round(Math.min(255, Math.max(0, newGray * ratioR)));
                    data[idx + 1] = Math.round(Math.min(255, Math.max(0, newGray * ratioG)));
                    data[idx + 2] = Math.round(Math.min(255, Math.max(0, newGray * ratioB)));
                }
            }
        }
        }
    }

    // 在控制台输出日志摘要
    alert('总块数:', log.length);
    alert('嵌入比特流 (前20位):', bits.slice(0, 20).join(''));
    alert('前5块详情:', log.slice(0, 5));
}

// ==================== 提取文字水印（投票冗余） ====================
function extractTextWatermark(processor, maxChars = 10) {
    const data = processor.imageData.data;
    const w = processor.width, h = processor.height;
    const totalBits = maxChars * 8;
    const votes = new Array(totalBits).fill(0);
    let blockCount = 0;

    for (let y = 0; y < h - 7; y += 8) {
        for (let x = 0; x < w - 7; x += 8) {
            const block = [];
            for (let dy = 0; dy < 8; dy++) {
                const row = [];
                for (let dx = 0; dx < 8; dx++) {
                    const idx = ((y + dy) * w + (x + dx)) * 4;
                    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                    row.push(rgbToGray(r, g, b));
                }
                block.push(row);
            }
            const bit = extractBitFromGrayBlock(block);
            const bitIndex = blockCount % totalBits;
            votes[bitIndex] += (bit === 1 ? 1 : -1);
            blockCount++;
        }
    }

    const resultBits = votes.map(v => v > 0 ? 1 : 0);
    return bitsToText(resultBits);
}
function debugExtract(processor, maxChars = 10) {
    const data = processor.imageData.data;
    const w = processor.width, h = processor.height;
    const totalBits = maxChars * 8;
    const votes = new Array(totalBits).fill(0);
    let blockCount = 0;
    const rawBits = [];

    for (let y = 0; y < h - 7; y += 8) {
        for (let x = 0; x < w - 7; x += 8) {
            const block = [];
            for (let dy = 0; dy < 8; dy++) {
                const row = [];
                for (let dx = 0; dx < 8; dx++) {
                    const idx = ((y + dy) * w + (x + dx)) * 4;
                    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                    row.push(rgbToGray(r, g, b));
                }
                block.push(row);
            }
            const bit = extractBitFromGrayBlock(block);
            const bitIndex = blockCount % totalBits;
            votes[bitIndex] += (bit === 1 ? 1 : -1);
            rawBits.push(bit);
            blockCount++;
        }
    }

    const resultBits = votes.map(v => v > 0 ? 1 : 0);
    const text = bitsToText(resultBits);
    alert('总可用块数:', blockCount);
    alert('需要的总比特数 (maxChars * 8):', totalBits);
    alert('投票数组 (前20个):', votes.slice(0, 20));
    alert('投票决策后的比特流 (前100位):', resultBits.slice(0, 100).join(''));
    alert('提取出的文字:', text);

    return { text, votes, resultBits, blockCount };
}
