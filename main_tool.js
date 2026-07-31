class ImageProcessor {
            /**
             * @param {string} imageSrc - 图片的 Data URL、Blob URL 或普通 URL 字符串
             */
            constructor(imageSrc) {
                this.src = imageSrc;
                this.imageData = null; // ImageData 对象，包含像素数据
                this.canvas = null; // 内部画布
                this.ctx = null; // 画布上下文
            }

            /**
             * 异步加载图片，完成后可进行像素操作
             * @returns {Promise<ImageProcessor>} 返回自身实例
             */
            load() {
                return new Promise((resolve, reject) => {
                    const img = new Image();
                    img.crossOrigin = 'anonymous'; // 处理跨域
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

            /**
             * 获取指定坐标的像素值（RGBA，范围 0-255）
             */
            getPixel(x, y) {
                if (!this.imageData) throw new Error('图片尚未加载，请先调用 load()');
                const idx = (y * this.imageData.width + x) * 4;
                const data = this.imageData.data;
                return {
                    r: data[idx],
                    g: data[idx + 1],
                    b: data[idx + 2],
                    a: data[idx + 3],
                };
            }

            /**
             * 设置指定坐标的像素值
             */
            setPixel(x, y, r, g, b, a = 255) {
                if (!this.imageData) throw new Error('图片尚未加载，请先调用 load()');
                const idx = (y * this.imageData.width + x) * 4;
                const data = this.imageData.data;
                data[idx] = r;
                data[idx + 1] = g;
                data[idx + 2] = b;
                data[idx + 3] = a;
            }

            /**
             * 遍历所有像素，回调函数接收当前像素对象和坐标，返回新像素对象（可选）
             * @param {function} callback - (pixel, x, y) => { r, g, b, a } 或 undefined
             */
            forEachPixel(callback) {
                if (!this.imageData) throw new Error('图片尚未加载，请先调用 load()');
                const data = this.imageData.data;
                const { width, height } = this.imageData;
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const idx = (y * width + x) * 4;
                        const pixel = {
                            r: data[idx],
                            g: data[idx + 1],
                            b: data[idx + 2],
                            a: data[idx + 3],
                        };
                        const newPixel = callback(pixel, x, y);
                        if (newPixel) {
                            data[idx] = newPixel.r ?? pixel.r;
                            data[idx + 1] = newPixel.g ?? pixel.g;
                            data[idx + 2] = newPixel.b ?? pixel.b;
                            data[idx + 3] = newPixel.a ?? pixel.a;
                        }
                    }
                }
            }

            /**
             * 将当前像素数据渲染到画布，并导出为 Data URL
             * @param {string} type - 图片类型，如 'image/png' 或 'image/jpeg'
             * @param {number} quality - JPEG 质量（0-1）
             * @returns {string} Data URL
             */
            toDataURL(type = 'image/png', quality = 0.92) {
                if (!this.ctx) throw new Error('图片尚未加载，请先调用 load()');
                this.ctx.putImageData(this.imageData, 0, 0);
                return this.canvas.toDataURL(type, quality);
            }

            /**
             * 获取图片宽度
             */
            get width() {
                return this.imageData?.width || 0;
            }

            /**
             * 获取图片高度
             */
            get height() {
                return this.imageData?.height || 0;
            }
        }
// 将文字转为二进制数组（UTF-8编码）
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

// 将二进制数组转回文字
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
// ---- 以下是 DCT 核心工具函数 ----

// 1D DCT (Type-II) 输入长度 8
function dct1d(input) {
    const N = 8;
    const output = new Float64Array(N);
    for (let k = 0; k < N; k++) {
        let sum = 0;
        for (let n = 0; n < N; n++) {
            sum += input[n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
        }
        const alpha = k === 0 ? 1 / Math.sqrt(2) : 1;
        output[k] = (2 / N) * alpha * sum;
    }
    return output;
}

// 1D IDCT (Type-III)
function idct1d(input) {
    const N = 8;
    const output = new Float64Array(N);
    for (let n = 0; n < N; n++) {
        let sum = 0;
        for (let k = 0; k < N; k++) {
            const alpha = k === 0 ? 1 / Math.sqrt(2) : 1;
            sum += alpha * input[k] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
        }
        output[n] = (2 / N) * sum;
    }
    return output;
}

// 2D DCT：对 8x8 矩阵的每行每列执行 1D DCT
function dct2d(matrix) {
    const N = 8;
    // 行变换
    const rows = matrix.map(row => dct1d(row));
    // 列变换
    const result = Array.from({ length: N }, (_, i) => {
        const col = rows.map(row => row[i]);
        return dct1d(col);
    });
    // 转置回去 (实际是列变换后得到新矩阵，但我们需要正确索引)
    const final = Array.from({ length: N }, (_, i) =>
        Array.from({ length: N }, (_, j) => result[j][i])
    );
    return final;
}

// 2D IDCT
function idct2d(matrix) {
    const N = 8;
    // 先对列做 IDCT
    const cols = Array.from({ length: N }, (_, i) => {
        const col = matrix.map(row => row[i]);
        return idct1d(col);
    });
    // 再对行做 IDCT
    const result = Array.from({ length: N }, (_, i) => {
        const row = cols.map(col => col[i]);
        return idct1d(row);
    });
    return result;
}

// ---- RGB <-> YCbCr 转换（只取 Y） ----
function rgbToY(r, g, b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

function yToRgb(y) {
    // 为了简化，假设 Cb=Cr=128（灰度），这样只修改亮度，不引入色偏
    const r = y + 1.402 * (128 - 128);
    const g = y - 0.344 * (128 - 128) - 0.714 * (128 - 128);
    const b = y + 1.772 * (128 - 128);
    return { r: Math.round(Math.min(255, Math.max(0, r))),
             g: Math.round(Math.min(255, Math.max(0, g))),
             b: Math.round(Math.min(255, Math.max(0, b))) };
}

// ---- 嵌入一个比特到 8x8 块的 Y 通道 ----
function embedBitInBlock(block, bit) {
    // 输入 block 是 8x8 的 Y 值 (0-255)
    // 1. DCT
    const dct = dct2d(block);
    // 2. 选择中频系数 (4,3) 或 (3,4) 等，避开直流 (0,0) 和高频
    const u = 4, v = 3;
    let coeff = dct[u][v];
    // 3. 根据 bit (0/1) 调整奇偶性
    const parity = Math.round(coeff) % 2;
    if (parity !== bit) {
        // 修改系数：如果当前是偶数但需要1，则加1；如果当前奇数但需要0，则减1
        coeff = Math.round(coeff) + (bit === 1 ? 1 : -1);
        dct[u][v] = coeff;
    }
    // 4. IDCT 回到空间域
    const idct = idct2d(dct);
    // 返回新的 Y 块 (四舍五入)
    return idct.map(row => row.map(val => Math.round(Math.min(255, Math.max(0, val)))));
}

// ---- 提取一个比特 ----
function extractBitFromBlock(block) {
    const dct = dct2d(block);
    const coeff = dct[4][3];
    return Math.round(coeff) % 2;
}
// 在图片中嵌入文字（每个块存一位，循环冗余）
function embedTextWatermark(processor, text) {
    const data = processor.imageData.data;
    const width = processor.width;
    const height = processor.height;
    const bits = textToBits(text);
    const bitLen = bits.length;

    let bitIndex = 0;
    for (let y = 0; y < height - 7; y += 8) {
        for (let x = 0; x < width - 7; x += 8) {
            // 提取该块的 Y 值
            const block = [];
            for (let dy = 0; dy < 8; dy++) {
                const row = [];
                for (let dx = 0; dx < 8; dx++) {
                    const idx = ((y + dy) * width + (x + dx)) * 4;
                    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                    row.push(0.299 * r + 0.587 * g + 0.114 * b);
                }
                block.push(row);
            }

            // 嵌入当前 bit（循环使用 bits）
            const bitToEmbed = bits[bitIndex % bitLen];
            const newBlock = embedBitInBlock(block, bitToEmbed);
            bitIndex++;

            // 写回 RGB
            for (let dy = 0; dy < 8; dy++) {
                for (let dx = 0; dx < 8; dx++) {
                    const idx = ((y + dy) * width + (x + dx)) * 4;
                    const yVal = newBlock[dy][dx];
                    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                    const cr = 0.713 * (r - yVal) + 128;
                    const cb = 0.564 * (b - yVal) + 128;
                    data[idx] = Math.round(Math.min(255, Math.max(0, yVal + 1.402 * (cr - 128))));
                    data[idx + 1] = Math.round(Math.min(255, Math.max(0, yVal - 0.344 * (cb - 128) - 0.714 * (cr - 128))));
                    data[idx + 2] = Math.round(Math.min(255, Math.max(0, yVal + 1.772 * (cb - 128))));
                }
            }
        }
    }
    processor.imageData.data = data;
}

// 从图片中提取文字（利用冗余投票机制）
function extractTextWatermark(processor, textLength) {
    const data = processor.imageData.data;
    const width = processor.width;
    const height = processor.height;
    const totalBits = textLength * 8; // 每个字符占8位（UTF-8）
    
    // 投票数组：记录每个bit位置被读出1的次数
    const votes = new Array(totalBits).fill(0);
    let blockCount = 0;

    for (let y = 0; y < height - 7; y += 8) {
        for (let x = 0; x < width - 7; x += 8) {
            // 提取该块的 Y 值
            const block = [];
            for (let dy = 0; dy < 8; dy++) {
                const row = [];
                for (let dx = 0; dx < 8; dx++) {
                    const idx = ((y + dy) * width + (x + dx)) * 4;
                    const r = data[idx], g = data[idx + 1], b = data[idx + 2];
                    row.push(0.299 * r + 0.587 * g + 0.114 * b);
                }
                block.push(row);
            }

            // 提取 DCT 系数奇偶性
            const dct = dct2d(block);
            const coeff = dct[4][3];
            const extractedBit = Math.round(coeff) % 2;
            
            // 投票：该块对应的是第几个 bit
            const bitIndex = blockCount % totalBits;
            if (extractedBit === 1) votes[bitIndex] += 1;
            else votes[bitIndex] -= 1; // 用减法来抵消0的权重
            
            blockCount++;
        }
    }

    // 根据投票结果还原 bits（大于0说明1多，小于0说明0多）
    const resultBits = votes.map(v => v > 0 ? 1 : 0);
    return bitsToText(resultBits);
}
