/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { complex } from '../ops/complex';
import { tensor } from '../ops/tensor';
import { sizeFromShape } from '../util';
import { DTYPE_VALUE_SIZE_MAP } from './types';
import { CompositeArrayBuffer } from './composite_array_buffer';
/** Number of bytes reserved for the length of the string. (32bit integer). */
const NUM_BYTES_STRING_LENGTH = 4;
/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @param group Group to which the weights belong (optional).
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
export async function encodeWeights(tensors, group) {
    // TODO(adarob, cais): Support quantization.
    const specs = [];
    const dataPromises = [];
    const names = Array.isArray(tensors) ?
        tensors.map(tensor => tensor.name) :
        Object.keys(tensors);
    for (let i = 0; i < names.length; ++i) {
        const name = names[i];
        const t = Array.isArray(tensors) ? tensors[i].tensor : tensors[name];
        if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool' &&
            t.dtype !== 'string' && t.dtype !== 'complex64') {
            throw new Error(`Unsupported dtype in weight '${name}': ${t.dtype}`);
        }
        const spec = { name, shape: t.shape, dtype: t.dtype };
        if (t.dtype === 'string') {
            const utf8bytes = new Promise(async (resolve) => {
                const vals = await t.bytes();
                const totalNumBytes = vals.reduce((p, c) => p + c.length, 0) +
                    NUM_BYTES_STRING_LENGTH * vals.length;
                const bytes = new Uint8Array(totalNumBytes);
                let offset = 0;
                for (let i = 0; i < vals.length; i++) {
                    const val = vals[i];
                    const bytesOfLength = new Uint8Array(new Uint32Array([val.length]).buffer);
                    bytes.set(bytesOfLength, offset);
                    offset += NUM_BYTES_STRING_LENGTH;
                    bytes.set(val, offset);
                    offset += val.length;
                }
                resolve(bytes);
            });
            dataPromises.push(utf8bytes);
        }
        else {
            dataPromises.push(t.data());
        }
        if (group != null) {
            spec.group = group;
        }
        specs.push(spec);
    }
    const tensorValues = await Promise.all(dataPromises);
    return { data: concatenateTypedArrays(tensorValues), specs };
}
/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param weightData A flat ArrayBuffer or an array of ArrayBuffers carrying the
 *   binary values of the tensors concatenated in the order specified in
 *   `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
export function decodeWeights(weightData, specs) {
    // TODO(adarob, cais): Support quantization.
    const compositeBuffer = new CompositeArrayBuffer(weightData);
    const out = {};
    let float16Decode;
    let offset = 0;
    for (const spec of specs) {
        const name = spec.name;
        const dtype = spec.dtype;
        const shape = spec.shape;
        const size = sizeFromShape(shape);
        let values;
        if ('quantization' in spec) {
            const quantization = spec.quantization;
            if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
                if (!('min' in quantization && 'scale' in quantization)) {
                    throw new Error(`Weight ${spec.name} with quantization ${quantization.dtype} ` +
                        `doesn't have corresponding metadata min and scale.`);
                }
            }
            else if (quantization.dtype === 'float16') {
                if (dtype !== 'float32') {
                    throw new Error(`Weight ${spec.name} is quantized with ${quantization.dtype} ` +
                        `which only supports weights of type float32 not ${dtype}.`);
                }
            }
            else {
                throw new Error(`Weight ${spec.name} has unknown ` +
                    `quantization dtype ${quantization.dtype}. ` +
                    `Supported quantization dtypes are: ` +
                    `'uint8', 'uint16', and 'float16'.`);
            }
            const quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
            const byteBuffer = compositeBuffer.slice(offset, offset + size * quantizationSizeFactor);
            const quantizedArray = (quantization.dtype === 'uint8') ?
                new Uint8Array(byteBuffer) :
                new Uint16Array(byteBuffer);
            if (dtype === 'float32') {
                if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
                    values = new Float32Array(quantizedArray.length);
                    for (let i = 0; i < quantizedArray.length; i++) {
                        const v = quantizedArray[i];
                        values[i] = v * quantization.scale + quantization.min;
                    }
                }
                else if (quantization.dtype === 'float16') {
                    if (float16Decode === undefined) {
                        float16Decode = getFloat16Decoder();
                    }
                    values = float16Decode(quantizedArray);
                }
                else {
                    throw new Error(`Unsupported quantization type ${quantization.dtype} ` +
                        `for weight type float32.`);
                }
            }
            else if (dtype === 'int32') {
                if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
                    throw new Error(`Unsupported quantization type ${quantization.dtype} ` +
                        `for weight type int32.`);
                }
                values = new Int32Array(quantizedArray.length);
                for (let i = 0; i < quantizedArray.length; i++) {
                    const v = quantizedArray[i];
                    values[i] = Math.round(v * quantization.scale + quantization.min);
                }
            }
            else {
                throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
            }
            offset += size * quantizationSizeFactor;
        }
        else if (dtype === 'string') {
            const size = sizeFromShape(spec.shape);
            values = [];
            for (let i = 0; i < size; i++) {
                const byteLength = new Uint32Array(compositeBuffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
                offset += NUM_BYTES_STRING_LENGTH;
                const bytes = new Uint8Array(compositeBuffer.slice(offset, offset + byteLength));
                values.push(bytes);
                offset += byteLength;
            }
        }
        else {
            const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
            const byteBuffer = compositeBuffer.slice(offset, offset + size * dtypeFactor);
            if (dtype === 'float32') {
                values = new Float32Array(byteBuffer);
            }
            else if (dtype === 'int32') {
                values = new Int32Array(byteBuffer);
            }
            else if (dtype === 'bool') {
                values = new Uint8Array(byteBuffer);
            }
            else if (dtype === 'complex64') {
                values = new Float32Array(byteBuffer);
                const real = new Float32Array(values.length / 2);
                const image = new Float32Array(values.length / 2);
                for (let i = 0; i < real.length; i++) {
                    real[i] = values[i * 2];
                    image[i] = values[i * 2 + 1];
                }
                const realTensor = tensor(real, shape, 'float32');
                const imageTensor = tensor(image, shape, 'float32');
                out[name] = complex(realTensor, imageTensor);
                realTensor.dispose();
                imageTensor.dispose();
            }
            else {
                throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
            }
            offset += size * dtypeFactor;
        }
        if (dtype !== 'complex64') {
            out[name] = tensor(values, shape, dtype);
        }
    }
    return out;
}
/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
export function concatenateTypedArrays(xs) {
    // TODO(adarob, cais): Support quantization.
    if (xs === null) {
        throw new Error(`Invalid input value: ${JSON.stringify(xs)}`);
    }
    let totalByteLength = 0;
    // `normalizedXs` is here for this reason: a `TypedArray`'s `buffer'
    // can have a different byte length from that of the `TypedArray` itself,
    // for example, when the `TypedArray` is created from an offset in an
    // `ArrayBuffer`. `normliazedXs` holds `TypedArray`s whose `buffer`s match
    // the `TypedArray` in byte length. If an element of `xs` does not show
    // this property, a new `TypedArray` that satisfy this property will be
    // constructed and pushed into `normalizedXs`.
    const normalizedXs = [];
    xs.forEach((x) => {
        totalByteLength += x.byteLength;
        // tslint:disable:no-any
        normalizedXs.push(x.byteLength === x.buffer.byteLength ? x :
            new x.constructor(x));
        if (!(x instanceof Float32Array || x instanceof Int32Array ||
            x instanceof Uint8Array)) {
            throw new Error(`Unsupported TypedArray subtype: ${x.constructor.name}`);
        }
        // tslint:enable:no-any
    });
    const y = new Uint8Array(totalByteLength);
    let offset = 0;
    normalizedXs.forEach((x) => {
        y.set(new Uint8Array(x.buffer), offset);
        offset += x.byteLength;
    });
    return y.buffer;
}
// Use Buffer on Node.js instead of Blob/atob/btoa
const useNodeBuffer = typeof Buffer !== 'undefined' &&
    (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
        typeof btoa === 'undefined');
/**
 * Calculate the byte length of a JavaScript string.
 *
 * Note that a JavaScript string can contain wide characters, therefore the
 * length of the string is not necessarily equal to the byte length.
 *
 * @param str Input string.
 * @returns Byte length.
 */
export function stringByteLength(str) {
    if (useNodeBuffer) {
        return Buffer.byteLength(str, 'utf8');
    }
    return new Blob([str]).size;
}
/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
export function arrayBufferToBase64String(buffer) {
    if (useNodeBuffer) {
        return Buffer.from(buffer).toString('base64');
    }
    const buf = new Uint8Array(buffer);
    let s = '';
    for (let i = 0, l = buf.length; i < l; i++) {
        s += String.fromCharCode(buf[i]);
    }
    return btoa(s);
}
/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
export function base64StringToArrayBuffer(str) {
    if (useNodeBuffer) {
        const buf = Buffer.from(str, 'base64');
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    const s = atob(str);
    const buffer = new Uint8Array(s.length);
    for (let i = 0; i < s.length; ++i) {
        buffer.set([s.charCodeAt(i)], i);
    }
    return buffer.buffer;
}
/**
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers An array of ArrayBuffers to concatenate, or a single
 *     ArrayBuffer.
 * @returns Result of concatenating `buffers` in order.
 *
 * @deprecated Use tf.io.CompositeArrayBuffer.join() instead.
 */
export function concatenateArrayBuffers(buffers) {
    return CompositeArrayBuffer.join(buffers);
}
/**
 * Get the basename of a path.
 *
 * Behaves in a way analogous to Linux's basename command.
 *
 * @param path
 */
export function basename(path) {
    const SEPARATOR = '/';
    path = path.trim();
    while (path.endsWith(SEPARATOR)) {
        path = path.slice(0, path.length - 1);
    }
    const items = path.split(SEPARATOR);
    return items[items.length - 1];
}
/**
 * Create `ModelJSON` from `ModelArtifacts`.
 *
 * @param artifacts Model artifacts, describing the model and its weights.
 * @param manifest Weight manifest, describing where the weights of the
 *     `ModelArtifacts` are stored, and some metadata about them.
 * @returns Object representing the `model.json` file describing the model
 *     artifacts and weights
 */
export function getModelJSONForModelArtifacts(artifacts, manifest) {
    const result = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: manifest
    };
    if (artifacts.signature != null) {
        result.signature = artifacts.signature;
    }
    if (artifacts.userDefinedMetadata != null) {
        result.userDefinedMetadata = artifacts.userDefinedMetadata;
    }
    if (artifacts.modelInitializer != null) {
        result.modelInitializer = artifacts.modelInitializer;
    }
    if (artifacts.initializerSignature != null) {
        result.initializerSignature = artifacts.initializerSignature;
    }
    if (artifacts.trainingConfig != null) {
        result.trainingConfig = artifacts.trainingConfig;
    }
    return result;
}
/**
 * Create `ModelArtifacts` from a JSON file and weights.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param weightSpecs The list of WeightsManifestEntry for the model. Must be
 *     passed if the modelJSON has a weightsManifest.
 * @param weightData An ArrayBuffer or array of ArrayBuffers of weight data for
 *     the model corresponding to the weights in weightSpecs. Must be passed if
 *     the modelJSON has a weightsManifest.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export function getModelArtifactsForJSONSync(modelJSON, weightSpecs, weightData) {
    const modelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy
    };
    if (modelJSON.trainingConfig != null) {
        modelArtifacts.trainingConfig = modelJSON.trainingConfig;
    }
    if (modelJSON.weightsManifest != null) {
        if (!weightSpecs) {
            throw new Error('modelJSON has weightsManifest but weightSpecs is null');
        }
        if (!weightData) {
            throw new Error('modelJSON has weightsManifest but weightData is null');
        }
        modelArtifacts.weightSpecs = weightSpecs;
        modelArtifacts.weightData = weightData;
    }
    if (modelJSON.signature != null) {
        modelArtifacts.signature = modelJSON.signature;
    }
    if (modelJSON.userDefinedMetadata != null) {
        modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
    }
    if (modelJSON.modelInitializer != null) {
        modelArtifacts.modelInitializer = modelJSON.modelInitializer;
    }
    if (modelJSON.initializerSignature != null) {
        modelArtifacts.initializerSignature = modelJSON.initializerSignature;
    }
    return modelArtifacts;
}
/**
 * Create `ModelArtifacts` from a JSON file.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param loadWeights Function that takes the JSON file's weights manifest,
 *     reads weights from the listed path(s), and returns a Promise of the
 *     weight manifest entries along with the weights data.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export async function getModelArtifactsForJSON(modelJSON, loadWeights) {
    let weightSpecs;
    let weightData;
    if (modelJSON.weightsManifest != null) {
        [weightSpecs, weightData] = await loadWeights(modelJSON.weightsManifest);
    }
    return getModelArtifactsForJSONSync(modelJSON, weightSpecs, weightData);
}
/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
export function getModelArtifactsInfoForJSON(modelArtifacts) {
    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
        throw new Error('Expected JSON model topology, received ArrayBuffer.');
    }
    return {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: modelArtifacts.modelTopology == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
        weightSpecsBytes: modelArtifacts.weightSpecs == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
        weightDataBytes: modelArtifacts.weightData == null ?
            0 :
            new CompositeArrayBuffer(modelArtifacts.weightData).byteLength,
    };
}
/**
 * Concatenate the weights stored in a WeightsManifestConfig into a list of
 * WeightsManifestEntry
 *
 * @param weightsManifest The WeightsManifestConfig to extract weights from.
 * @returns A list of WeightsManifestEntry of the weights in the weightsManifest
 */
export function getWeightSpecs(weightsManifest) {
    const weightSpecs = [];
    for (const entry of weightsManifest) {
        weightSpecs.push(...entry.weights);
    }
    return weightSpecs;
}
/**
 * Computes mantisa table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 2048 mantissa lookup values.
 */
function computeFloat16MantisaTable() {
    const convertMantissa = (i) => {
        let m = i << 13;
        let e = 0;
        while ((m & 0x00800000) === 0) {
            e -= 0x00800000;
            m <<= 1;
        }
        m &= ~0x00800000;
        e += 0x38800000;
        return m | e;
    };
    const mantisaTable = new Uint32Array(2048);
    mantisaTable[0] = 0;
    for (let i = 1; i < 1024; i++) {
        mantisaTable[i] = convertMantissa(i);
    }
    for (let i = 1024; i < 2048; i++) {
        mantisaTable[i] = 0x38000000 + ((i - 1024) << 13);
    }
    return mantisaTable;
}
/**
 * Computes exponent table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 64 exponent lookup values.
 */
function computeFloat16ExponentTable() {
    const exponentTable = new Uint32Array(64);
    exponentTable[0] = 0;
    exponentTable[31] = 0x47800000;
    exponentTable[32] = 0x80000000;
    exponentTable[63] = 0xc7800000;
    for (let i = 1; i < 31; i++) {
        exponentTable[i] = i << 23;
    }
    for (let i = 33; i < 63; i++) {
        exponentTable[i] = 0x80000000 + ((i - 32) << 23);
    }
    return exponentTable;
}
/**
 * Computes offset table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 6d offset values.
 */
function computeFloat16OffsetTable() {
    const offsetTable = new Uint32Array(64);
    for (let i = 0; i < 64; i++) {
        offsetTable[i] = 1024;
    }
    offsetTable[0] = offsetTable[32] = 0;
    return offsetTable;
}
/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
export function getFloat16Decoder() {
    // Algorithm is based off of
    // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
    // Cache lookup tables
    const mantisaTable = computeFloat16MantisaTable();
    const exponentTable = computeFloat16ExponentTable();
    const offsetTable = computeFloat16OffsetTable();
    return (quantizedArray) => {
        const buffer = new ArrayBuffer(4 * quantizedArray.length);
        const bufferUint32View = new Uint32Array(buffer);
        for (let index = 0; index < quantizedArray.length; index++) {
            const float16Bits = quantizedArray[index];
            const float32Bits = mantisaTable[offsetTable[float16Bits >> 10] + (float16Bits & 0x3ff)] +
                exponentTable[float16Bits >> 10];
            bufferUint32View[index] = float32Bits;
        }
        return new Float32Array(buffer);
    };
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW9fdXRpbHMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL2lvL2lvX3V0aWxzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUN2QyxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBR3JDLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFdEMsT0FBTyxFQUFDLG9CQUFvQixFQUFzSCxNQUFNLFNBQVMsQ0FBQztBQUNsSyxPQUFPLEVBQUMsb0JBQW9CLEVBQUMsTUFBTSwwQkFBMEIsQ0FBQztBQUU5RCw4RUFBOEU7QUFDOUUsTUFBTSx1QkFBdUIsR0FBRyxDQUFDLENBQUM7QUFFbEM7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnQkc7QUFDSCxNQUFNLENBQUMsS0FBSyxVQUFVLGFBQWEsQ0FDL0IsT0FBcUMsRUFBRSxLQUFtQjtJQUU1RCw0Q0FBNEM7SUFDNUMsTUFBTSxLQUFLLEdBQTJCLEVBQUUsQ0FBQztJQUN6QyxNQUFNLFlBQVksR0FBK0IsRUFBRSxDQUFDO0lBRXBELE1BQU0sS0FBSyxHQUFhLEtBQUssQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUM1QyxPQUFPLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7UUFDcEMsTUFBTSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztJQUV6QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsS0FBSyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtRQUNyQyxNQUFNLElBQUksR0FBRyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdEIsTUFBTSxDQUFDLEdBQUcs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ZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2NvbXBsZXh9IGZyb20gJy4uL29wcy9jb21wbGV4JztcbmltcG9ydCB7dGVuc29yfSBmcm9tICcuLi9vcHMvdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3IsIE5hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtUeXBlZEFycmF5fSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge3NpemVGcm9tU2hhcGV9IGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge0RUWVBFX1ZBTFVFX1NJWkVfTUFQLCBNb2RlbEFydGlmYWN0cywgTW9kZWxBcnRpZmFjdHNJbmZvLCBNb2RlbEpTT04sIFdlaWdodERhdGEsIFdlaWdodEdyb3VwLCBXZWlnaHRzTWFuaWZlc3RDb25maWcsIFdlaWdodHNNYW5pZmVzdEVudHJ5fSBmcm9tICcuL3R5cGVzJztcbmltcG9ydCB7Q29tcG9zaXRlQXJyYXlCdWZmZXJ9IGZyb20gJy4vY29tcG9zaXRlX2FycmF5X2J1ZmZlcic7XG5cbi8qKiBOdW1iZXIgb2YgYnl0ZXMgcmVzZXJ2ZWQgZm9yIHRoZSBsZW5ndGggb2YgdGhlIHN0cmluZy4gKDMyYml0IGludGVnZXIpLiAqL1xuY29uc3QgTlVNX0JZVEVTX1NUUklOR19MRU5HVEggPSA0O1xuXG4vKipcbiAqIEVuY29kZSBhIG1hcCBmcm9tIG5hbWVzIHRvIHdlaWdodCB2YWx1ZXMgYXMgYW4gQXJyYXlCdWZmZXIsIGFsb25nIHdpdGggYW5cbiAqIGBBcnJheWAgb2YgYFdlaWdodHNNYW5pZmVzdEVudHJ5YCBhcyBzcGVjaWZpY2F0aW9uIG9mIHRoZSBlbmNvZGVkIHdlaWdodHMuXG4gKlxuICogVGhpcyBmdW5jdGlvbiBkb2VzIG5vdCBwZXJmb3JtIHNoYXJkaW5nLlxuICpcbiAqIFRoaXMgZnVuY3Rpb24gaXMgdGhlIHJldmVyc2Ugb2YgYGRlY29kZVdlaWdodHNgLlxuICpcbiAqIEBwYXJhbSB0ZW5zb3JzIEEgbWFwIChcImRpY3RcIikgZnJvbSBuYW1lcyB0byB0ZW5zb3JzLlxuICogQHBhcmFtIGdyb3VwIEdyb3VwIHRvIHdoaWNoIHRoZSB3ZWlnaHRzIGJlbG9uZyAob3B0aW9uYWwpLlxuICogQHJldHVybnMgQSBgUHJvbWlzZWAgb2ZcbiAqICAgLSBBIGZsYXQgYEFycmF5QnVmZmVyYCB3aXRoIGFsbCB0aGUgYmluYXJ5IHZhbHVlcyBvZiB0aGUgYFRlbnNvcmBzXG4gKiAgICAgY29uY2F0ZW5hdGVkLlxuICogICAtIEFuIGBBcnJheWAgb2YgYFdlaWdodE1hbmlmZXN0RW50cnlgcywgY2FycnlpbmcgaW5mb3JtYXRpb24gaW5jbHVkaW5nXG4gKiAgICAgdGVuc29yIG5hbWVzLCBgZHR5cGVgcyBhbmQgc2hhcGVzLlxuICogQHRocm93cyBFcnJvcjogb24gdW5zdXBwb3J0ZWQgdGVuc29yIGBkdHlwZWAuXG4gKi9cbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBlbmNvZGVXZWlnaHRzKFxuICAgIHRlbnNvcnM6IE5hbWVkVGVuc29yTWFwfE5hbWVkVGVuc29yW10sIGdyb3VwPzogV2VpZ2h0R3JvdXApOlxuICAgIFByb21pc2U8e2RhdGE6IEFycmF5QnVmZmVyLCBzcGVjczogV2VpZ2h0c01hbmlmZXN0RW50cnlbXX0+IHtcbiAgLy8gVE9ETyhhZGFyb2IsIGNhaXMpOiBTdXBwb3J0IHF1YW50aXphdGlvbi5cbiAgY29uc3Qgc3BlY3M6IFdlaWdodHNNYW5pZmVzdEVudHJ5W10gPSBbXTtcbiAgY29uc3QgZGF0YVByb21pc2VzOiBBcnJheTxQcm9taXNlPFR5cGVkQXJyYXk+PiA9IFtdO1xuXG4gIGNvbnN0IG5hbWVzOiBzdHJpbmdbXSA9IEFycmF5LmlzQXJyYXkodGVuc29ycykgP1xuICAgICAgdGVuc29ycy5tYXAodGVuc29yID0+IHRlbnNvci5uYW1lKSA6XG4gICAgICBPYmplY3Qua2V5cyh0ZW5zb3JzKTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IG5hbWVzLmxlbmd0aDsgKytpKSB7XG4gICAgY29uc3QgbmFtZSA9IG5hbWVzW2ldO1xuICAgIGNvbnN0IHQgPSBBcnJheS5pc0FycmF5KHRlbnNvcnMpID8gdGVuc29yc1tpXS50ZW5zb3IgOiB0ZW5zb3JzW25hbWVdO1xuICAgIGlmICh0LmR0eXBlICE9PSAnZmxvYXQzMicgJiYgdC5kdHlwZSAhPT0gJ2ludDMyJyAmJiB0LmR0eXBlICE9PSAnYm9vbCcgJiZcbiAgICAgICAgdC5kdHlwZSAhPT0gJ3N0cmluZycgJiYgdC5kdHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVW5zdXBwb3J0ZWQgZHR5cGUgaW4gd2VpZ2h0ICcke25hbWV9JzogJHt0LmR0eXBlfWApO1xuICAgIH1cbiAgICBjb25zdCBzcGVjOiBXZWlnaHRzTWFuaWZlc3RFbnRyeSA9IHtuYW1lLCBzaGFwZTogdC5zaGFwZSwgZHR5cGU6IHQuZHR5cGV9O1xuICAgIGlmICh0LmR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgY29uc3QgdXRmOGJ5dGVzID0gbmV3IFByb21pc2U8VHlwZWRBcnJheT4oYXN5bmMgcmVzb2x2ZSA9PiB7XG4gICAgICAgIGNvbnN0IHZhbHMgPSBhd2FpdCB0LmJ5dGVzKCkgYXMgVWludDhBcnJheVtdO1xuICAgICAgICBjb25zdCB0b3RhbE51bUJ5dGVzID0gdmFscy5yZWR1Y2UoKHAsIGMpID0+IHAgKyBjLmxlbmd0aCwgMCkgK1xuICAgICAgICAgICAgTlVNX0JZVEVTX1NUUklOR19MRU5HVEggKiB2YWxzLmxlbmd0aDtcbiAgICAgICAgY29uc3QgYnl0ZXMgPSBuZXcgVWludDhBcnJheSh0b3RhbE51bUJ5dGVzKTtcbiAgICAgICAgbGV0IG9mZnNldCA9IDA7XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFscy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgIGNvbnN0IHZhbCA9IHZhbHNbaV07XG4gICAgICAgICAgY29uc3QgYnl0ZXNPZkxlbmd0aCA9XG4gICAgICAgICAgICAgIG5ldyBVaW50OEFycmF5KG5ldyBVaW50MzJBcnJheShbdmFsLmxlbmd0aF0pLmJ1ZmZlcik7XG4gICAgICAgICAgYnl0ZXMuc2V0KGJ5dGVzT2ZMZW5ndGgsIG9mZnNldCk7XG4gICAgICAgICAgb2Zmc2V0ICs9IE5VTV9CWVRFU19TVFJJTkdfTEVOR1RIO1xuICAgICAgICAgIGJ5dGVzLnNldCh2YWwsIG9mZnNldCk7XG4gICAgICAgICAgb2Zmc2V0ICs9IHZhbC5sZW5ndGg7XG4gICAgICAgIH1cbiAgICAgICAgcmVzb2x2ZShieXRlcyk7XG4gICAgICB9KTtcbiAgICAgIGRhdGFQcm9taXNlcy5wdXNoKHV0ZjhieXRlcyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGRhdGFQcm9taXNlcy5wdXNoKHQuZGF0YSgpKTtcbiAgICB9XG4gICAgaWYgKGdyb3VwICE9IG51bGwpIHtcbiAgICAgIHNwZWMuZ3JvdXAgPSBncm91cDtcbiAgICB9XG4gICAgc3BlY3MucHVzaChzcGVjKTtcbiAgfVxuXG4gIGNvbnN0IHRlbnNvclZhbHVlcyA9IGF3YWl0IFByb21pc2UuYWxsKGRhdGFQcm9taXNlcyk7XG4gIHJldHVybiB7ZGF0YTogY29uY2F0ZW5hdGVUeXBlZEFycmF5cyh0ZW5zb3JWYWx1ZXMpLCBzcGVjc307XG59XG5cbi8qKlxuICogRGVjb2RlIGZsYXQgQXJyYXlCdWZmZXIgYXMgd2VpZ2h0cy5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIGRvZXMgbm90IGhhbmRsZSBzaGFyZGluZy5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIGlzIHRoZSByZXZlcnNlIG9mIGBlbmNvZGVXZWlnaHRzYC5cbiAqXG4gKiBAcGFyYW0gd2VpZ2h0RGF0YSBBIGZsYXQgQXJyYXlCdWZmZXIgb3IgYW4gYXJyYXkgb2YgQXJyYXlCdWZmZXJzIGNhcnJ5aW5nIHRoZVxuICogICBiaW5hcnkgdmFsdWVzIG9mIHRoZSB0ZW5zb3JzIGNvbmNhdGVuYXRlZCBpbiB0aGUgb3JkZXIgc3BlY2lmaWVkIGluXG4gKiAgIGBzcGVjc2AuXG4gKiBAcGFyYW0gc3BlY3MgU3BlY2lmaWNhdGlvbnMgb2YgdGhlIG5hbWVzLCBkdHlwZXMgYW5kIHNoYXBlcyBvZiB0aGUgdGVuc29yc1xuICogICB3aG9zZSB2YWx1ZSBhcmUgZW5jb2RlZCBieSBgYnVmZmVyYC5cbiAqIEByZXR1cm4gQSBtYXAgZnJvbSB0ZW5zb3IgbmFtZSB0byB0ZW5zb3IgdmFsdWUsIHdpdGggdGhlIG5hbWVzIGNvcnJlc3BvbmRpbmdcbiAqICAgdG8gbmFtZXMgaW4gYHNwZWNzYC5cbiAqIEB0aHJvd3MgRXJyb3IsIGlmIGFueSBvZiB0aGUgdGVuc29ycyBoYXMgdW5zdXBwb3J0ZWQgZHR5cGUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBkZWNvZGVXZWlnaHRzKFxuICAgIHdlaWdodERhdGE6IFdlaWdodERhdGEsXG4gICAgc3BlY3M6IFdlaWdodHNNYW5pZmVzdEVudHJ5W10pOiBOYW1lZFRlbnNvck1hcCB7XG4gIC8vIFRPRE8oYWRhcm9iLCBjYWlzKTogU3VwcG9ydCBxdWFudGl6YXRpb24uXG4gIGNvbnN0IGNvbXBvc2l0ZUJ1ZmZlciA9IG5ldyBDb21wb3NpdGVBcnJheUJ1ZmZlcih3ZWlnaHREYXRhKTtcbiAgY29uc3Qgb3V0OiBOYW1lZFRlbnNvck1hcCA9IHt9O1xuICBsZXQgZmxvYXQxNkRlY29kZTogKGJ1ZmZlcjogVWludDE2QXJyYXkpID0+IEZsb2F0MzJBcnJheSB8IHVuZGVmaW5lZDtcbiAgbGV0IG9mZnNldCA9IDA7XG4gIGZvciAoY29uc3Qgc3BlYyBvZiBzcGVjcykge1xuICAgIGNvbnN0IG5hbWUgPSBzcGVjLm5hbWU7XG4gICAgY29uc3QgZHR5cGUgPSBzcGVjLmR0eXBlO1xuICAgIGNvbnN0IHNoYXBlID0gc3BlYy5zaGFwZTtcbiAgICBjb25zdCBzaXplID0gc2l6ZUZyb21TaGFwZShzaGFwZSk7XG4gICAgbGV0IHZhbHVlczogVHlwZWRBcnJheXxzdHJpbmdbXXxVaW50OEFycmF5W107XG5cbiAgICBpZiAoJ3F1YW50aXphdGlvbicgaW4gc3BlYykge1xuICAgICAgY29uc3QgcXVhbnRpemF0aW9uID0gc3BlYy5xdWFudGl6YXRpb247XG4gICAgICBpZiAocXVhbnRpemF0aW9uLmR0eXBlID09PSAndWludDgnIHx8IHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ3VpbnQxNicpIHtcbiAgICAgICAgaWYgKCEoJ21pbicgaW4gcXVhbnRpemF0aW9uICYmICdzY2FsZScgaW4gcXVhbnRpemF0aW9uKSkge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFdlaWdodCAke3NwZWMubmFtZX0gd2l0aCBxdWFudGl6YXRpb24gJHtxdWFudGl6YXRpb24uZHR5cGV9IGAgK1xuICAgICAgICAgICAgICBgZG9lc24ndCBoYXZlIGNvcnJlc3BvbmRpbmcgbWV0YWRhdGEgbWluIGFuZCBzY2FsZS5gKTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIGlmIChxdWFudGl6YXRpb24uZHR5cGUgPT09ICdmbG9hdDE2Jykge1xuICAgICAgICBpZiAoZHR5cGUgIT09ICdmbG9hdDMyJykge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFdlaWdodCAke3NwZWMubmFtZX0gaXMgcXVhbnRpemVkIHdpdGggJHtxdWFudGl6YXRpb24uZHR5cGV9IGAgK1xuICAgICAgICAgICAgICBgd2hpY2ggb25seSBzdXBwb3J0cyB3ZWlnaHRzIG9mIHR5cGUgZmxvYXQzMiBub3QgJHtkdHlwZX0uYCk7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgIGBXZWlnaHQgJHtzcGVjLm5hbWV9IGhhcyB1bmtub3duIGAgK1xuICAgICAgICAgICAgYHF1YW50aXphdGlvbiBkdHlwZSAke3F1YW50aXphdGlvbi5kdHlwZX0uIGAgK1xuICAgICAgICAgICAgYFN1cHBvcnRlZCBxdWFudGl6YXRpb24gZHR5cGVzIGFyZTogYCArXG4gICAgICAgICAgICBgJ3VpbnQ4JywgJ3VpbnQxNicsIGFuZCAnZmxvYXQxNicuYCk7XG4gICAgICB9XG4gICAgICBjb25zdCBxdWFudGl6YXRpb25TaXplRmFjdG9yID0gRFRZUEVfVkFMVUVfU0laRV9NQVBbcXVhbnRpemF0aW9uLmR0eXBlXTtcbiAgICAgIGNvbnN0IGJ5dGVCdWZmZXIgPVxuICAgICAgICAgIGNvbXBvc2l0ZUJ1ZmZlci5zbGljZShvZmZzZXQsIG9mZnNldCArIHNpemUgKiBxdWFudGl6YXRpb25TaXplRmFjdG9yKTtcbiAgICAgIGNvbnN0IHF1YW50aXplZEFycmF5ID0gKHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ3VpbnQ4JykgP1xuICAgICAgICAgIG5ldyBVaW50OEFycmF5KGJ5dGVCdWZmZXIpIDpcbiAgICAgICAgICBuZXcgVWludDE2QXJyYXkoYnl0ZUJ1ZmZlcik7XG4gICAgICBpZiAoZHR5cGUgPT09ICdmbG9hdDMyJykge1xuICAgICAgICBpZiAocXVhbnRpemF0aW9uLmR0eXBlID09PSAndWludDgnIHx8IHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ3VpbnQxNicpIHtcbiAgICAgICAgICB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHF1YW50aXplZEFycmF5Lmxlbmd0aCk7XG4gICAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBxdWFudGl6ZWRBcnJheS5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgY29uc3QgdiA9IHF1YW50aXplZEFycmF5W2ldO1xuICAgICAgICAgICAgdmFsdWVzW2ldID0gdiAqIHF1YW50aXphdGlvbi5zY2FsZSArIHF1YW50aXphdGlvbi5taW47XG4gICAgICAgICAgfVxuICAgICAgICB9IGVsc2UgaWYgKHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ2Zsb2F0MTYnKSB7XG4gICAgICAgICAgaWYgKGZsb2F0MTZEZWNvZGUgPT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgZmxvYXQxNkRlY29kZSA9IGdldEZsb2F0MTZEZWNvZGVyKCk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHZhbHVlcyA9IGZsb2F0MTZEZWNvZGUocXVhbnRpemVkQXJyYXkgYXMgVWludDE2QXJyYXkpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFVuc3VwcG9ydGVkIHF1YW50aXphdGlvbiB0eXBlICR7cXVhbnRpemF0aW9uLmR0eXBlfSBgICtcbiAgICAgICAgICAgICAgYGZvciB3ZWlnaHQgdHlwZSBmbG9hdDMyLmApO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgaWYgKGR0eXBlID09PSAnaW50MzInKSB7XG4gICAgICAgIGlmIChxdWFudGl6YXRpb24uZHR5cGUgIT09ICd1aW50OCcgJiYgcXVhbnRpemF0aW9uLmR0eXBlICE9PSAndWludDE2Jykge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFVuc3VwcG9ydGVkIHF1YW50aXphdGlvbiB0eXBlICR7cXVhbnRpemF0aW9uLmR0eXBlfSBgICtcbiAgICAgICAgICAgICAgYGZvciB3ZWlnaHQgdHlwZSBpbnQzMi5gKTtcbiAgICAgICAgfVxuICAgICAgICB2YWx1ZXMgPSBuZXcgSW50MzJBcnJheShxdWFudGl6ZWRBcnJheS5sZW5ndGgpO1xuICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHF1YW50aXplZEFycmF5Lmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgY29uc3QgdiA9IHF1YW50aXplZEFycmF5W2ldO1xuICAgICAgICAgIHZhbHVlc1tpXSA9IE1hdGgucm91bmQodiAqIHF1YW50aXphdGlvbi5zY2FsZSArIHF1YW50aXphdGlvbi5taW4pO1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoYFVuc3VwcG9ydGVkIGR0eXBlIGluIHdlaWdodCAnJHtuYW1lfSc6ICR7ZHR5cGV9YCk7XG4gICAgICB9XG4gICAgICBvZmZzZXQgKz0gc2l6ZSAqIHF1YW50aXphdGlvblNpemVGYWN0b3I7XG4gICAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIGNvbnN0IHNpemUgPSBzaXplRnJvbVNoYXBlKHNwZWMuc2hhcGUpO1xuICAgICAgdmFsdWVzID0gW107XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHNpemU7IGkrKykge1xuICAgICAgICBjb25zdCBieXRlTGVuZ3RoID0gbmV3IFVpbnQzMkFycmF5KFxuICAgICAgICAgICAgY29tcG9zaXRlQnVmZmVyLnNsaWNlKG9mZnNldCwgb2Zmc2V0ICsgTlVNX0JZVEVTX1NUUklOR19MRU5HVEgpKVswXTtcbiAgICAgICAgb2Zmc2V0ICs9IE5VTV9CWVRFU19TVFJJTkdfTEVOR1RIO1xuICAgICAgICBjb25zdCBieXRlcyA9IG5ldyBVaW50OEFycmF5KFxuICAgICAgICAgIGNvbXBvc2l0ZUJ1ZmZlci5zbGljZShvZmZzZXQsIG9mZnNldCArIGJ5dGVMZW5ndGgpKTtcbiAgICAgICAgKHZhbHVlcyBhcyBVaW50OEFycmF5W10pLnB1c2goYnl0ZXMpO1xuICAgICAgICBvZmZzZXQgKz0gYnl0ZUxlbmd0aDtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgZHR5cGVGYWN0b3IgPSBEVFlQRV9WQUxVRV9TSVpFX01BUFtkdHlwZV07XG4gICAgICBjb25zdCBieXRlQnVmZmVyID0gY29tcG9zaXRlQnVmZmVyLnNsaWNlKG9mZnNldCxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb2Zmc2V0ICsgc2l6ZSAqIGR0eXBlRmFjdG9yKTtcblxuICAgICAgaWYgKGR0eXBlID09PSAnZmxvYXQzMicpIHtcbiAgICAgICAgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShieXRlQnVmZmVyKTtcbiAgICAgIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAgICAgICAgdmFsdWVzID0gbmV3IEludDMyQXJyYXkoYnl0ZUJ1ZmZlcik7XG4gICAgICB9IGVsc2UgaWYgKGR0eXBlID09PSAnYm9vbCcpIHtcbiAgICAgICAgdmFsdWVzID0gbmV3IFVpbnQ4QXJyYXkoYnl0ZUJ1ZmZlcik7XG4gICAgICB9IGVsc2UgaWYgKGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgICB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGJ5dGVCdWZmZXIpO1xuICAgICAgICBjb25zdCByZWFsID0gbmV3IEZsb2F0MzJBcnJheSh2YWx1ZXMubGVuZ3RoIC8gMik7XG4gICAgICAgIGNvbnN0IGltYWdlID0gbmV3IEZsb2F0MzJBcnJheSh2YWx1ZXMubGVuZ3RoIC8gMik7XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcmVhbC5sZW5ndGg7IGkrKykge1xuICAgICAgICAgIHJlYWxbaV0gPSB2YWx1ZXNbaSAqIDJdO1xuICAgICAgICAgIGltYWdlW2ldID0gdmFsdWVzW2kgKiAyICsgMV07XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgcmVhbFRlbnNvciA9IHRlbnNvcihyZWFsLCBzaGFwZSwgJ2Zsb2F0MzInKTtcbiAgICAgICAgY29uc3QgaW1hZ2VUZW5zb3IgPSB0ZW5zb3IoaW1hZ2UsIHNoYXBlLCAnZmxvYXQzMicpO1xuICAgICAgICBvdXRbbmFtZV0gPSBjb21wbGV4KHJlYWxUZW5zb3IsIGltYWdlVGVuc29yKTtcbiAgICAgICAgcmVhbFRlbnNvci5kaXNwb3NlKCk7XG4gICAgICAgIGltYWdlVGVuc29yLmRpc3Bvc2UoKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihgVW5zdXBwb3J0ZWQgZHR5cGUgaW4gd2VpZ2h0ICcke25hbWV9JzogJHtkdHlwZX1gKTtcbiAgICAgIH1cbiAgICAgIG9mZnNldCArPSBzaXplICogZHR5cGVGYWN0b3I7XG4gICAgfVxuICAgIGlmIChkdHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIG91dFtuYW1lXSA9IHRlbnNvcih2YWx1ZXMsIHNoYXBlLCBkdHlwZSk7XG4gICAgfVxuICB9XG4gIHJldHVybiBvdXQ7XG59XG5cbi8qKlxuICogQ29uY2F0ZW5hdGUgVHlwZWRBcnJheXMgaW50byBhbiBBcnJheUJ1ZmZlci5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbmNhdGVuYXRlVHlwZWRBcnJheXMoeHM6IFR5cGVkQXJyYXlbXSk6IEFycmF5QnVmZmVyIHtcbiAgLy8gVE9ETyhhZGFyb2IsIGNhaXMpOiBTdXBwb3J0IHF1YW50aXphdGlvbi5cbiAgaWYgKHhzID09PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBJbnZhbGlkIGlucHV0IHZhbHVlOiAke0pTT04uc3RyaW5naWZ5KHhzKX1gKTtcbiAgfVxuXG4gIGxldCB0b3RhbEJ5dGVMZW5ndGggPSAwO1xuXG4gIC8vIGBub3JtYWxpemVkWHNgIGlzIGhlcmUgZm9yIHRoaXMgcmVhc29uOiBhIGBUeXBlZEFycmF5YCdzIGBidWZmZXInXG4gIC8vIGNhbiBoYXZlIGEgZGlmZmVyZW50IGJ5dGUgbGVuZ3RoIGZyb20gdGhhdCBvZiB0aGUgYFR5cGVkQXJyYXlgIGl0c2VsZixcbiAgLy8gZm9yIGV4YW1wbGUsIHdoZW4gdGhlIGBUeXBlZEFycmF5YCBpcyBjcmVhdGVkIGZyb20gYW4gb2Zmc2V0IGluIGFuXG4gIC8vIGBBcnJheUJ1ZmZlcmAuIGBub3JtbGlhemVkWHNgIGhvbGRzIGBUeXBlZEFycmF5YHMgd2hvc2UgYGJ1ZmZlcmBzIG1hdGNoXG4gIC8vIHRoZSBgVHlwZWRBcnJheWAgaW4gYnl0ZSBsZW5ndGguIElmIGFuIGVsZW1lbnQgb2YgYHhzYCBkb2VzIG5vdCBzaG93XG4gIC8vIHRoaXMgcHJvcGVydHksIGEgbmV3IGBUeXBlZEFycmF5YCB0aGF0IHNhdGlzZnkgdGhpcyBwcm9wZXJ0eSB3aWxsIGJlXG4gIC8vIGNvbnN0cnVjdGVkIGFuZCBwdXNoZWQgaW50byBgbm9ybWFsaXplZFhzYC5cbiAgY29uc3Qgbm9ybWFsaXplZFhzOiBUeXBlZEFycmF5W10gPSBbXTtcbiAgeHMuZm9yRWFjaCgoeDogVHlwZWRBcnJheSkgPT4ge1xuICAgIHRvdGFsQnl0ZUxlbmd0aCArPSB4LmJ5dGVMZW5ndGg7XG4gICAgLy8gdHNsaW50OmRpc2FibGU6bm8tYW55XG4gICAgbm9ybWFsaXplZFhzLnB1c2goXG4gICAgICAgIHguYnl0ZUxlbmd0aCA9PT0geC5idWZmZXIuYnl0ZUxlbmd0aCA/IHggOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXcgKHguY29uc3RydWN0b3IgYXMgYW55KSh4KSk7XG4gICAgaWYgKCEoeCBhcyBhbnkgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkgfHwgeCBhcyBhbnkgaW5zdGFuY2VvZiBJbnQzMkFycmF5IHx8XG4gICAgICAgICAgeCBhcyBhbnkgaW5zdGFuY2VvZiBVaW50OEFycmF5KSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBVbnN1cHBvcnRlZCBUeXBlZEFycmF5IHN1YnR5cGU6ICR7eC5jb25zdHJ1Y3Rvci5uYW1lfWApO1xuICAgIH1cbiAgICAvLyB0c2xpbnQ6ZW5hYmxlOm5vLWFueVxuICB9KTtcblxuICBjb25zdCB5ID0gbmV3IFVpbnQ4QXJyYXkodG90YWxCeXRlTGVuZ3RoKTtcbiAgbGV0IG9mZnNldCA9IDA7XG4gIG5vcm1hbGl6ZWRYcy5mb3JFYWNoKCh4OiBUeXBlZEFycmF5KSA9PiB7XG4gICAgeS5zZXQobmV3IFVpbnQ4QXJyYXkoeC5idWZmZXIpLCBvZmZzZXQpO1xuICAgIG9mZnNldCArPSB4LmJ5dGVMZW5ndGg7XG4gIH0pO1xuXG4gIHJldHVybiB5LmJ1ZmZlcjtcbn1cblxuLy8gVXNlIEJ1ZmZlciBvbiBOb2RlLmpzIGluc3RlYWQgb2YgQmxvYi9hdG9iL2J0b2FcbmNvbnN0IHVzZU5vZGVCdWZmZXIgPSB0eXBlb2YgQnVmZmVyICE9PSAndW5kZWZpbmVkJyAmJlxuICAgICh0eXBlb2YgQmxvYiA9PT0gJ3VuZGVmaW5lZCcgfHwgdHlwZW9mIGF0b2IgPT09ICd1bmRlZmluZWQnIHx8XG4gICAgIHR5cGVvZiBidG9hID09PSAndW5kZWZpbmVkJyk7XG5cbi8qKlxuICogQ2FsY3VsYXRlIHRoZSBieXRlIGxlbmd0aCBvZiBhIEphdmFTY3JpcHQgc3RyaW5nLlxuICpcbiAqIE5vdGUgdGhhdCBhIEphdmFTY3JpcHQgc3RyaW5nIGNhbiBjb250YWluIHdpZGUgY2hhcmFjdGVycywgdGhlcmVmb3JlIHRoZVxuICogbGVuZ3RoIG9mIHRoZSBzdHJpbmcgaXMgbm90IG5lY2Vzc2FyaWx5IGVxdWFsIHRvIHRoZSBieXRlIGxlbmd0aC5cbiAqXG4gKiBAcGFyYW0gc3RyIElucHV0IHN0cmluZy5cbiAqIEByZXR1cm5zIEJ5dGUgbGVuZ3RoLlxuICovXG5leHBvcnQgZnVuY3Rpb24gc3RyaW5nQnl0ZUxlbmd0aChzdHI6IHN0cmluZyk6IG51bWJlciB7XG4gIGlmICh1c2VOb2RlQnVmZmVyKSB7XG4gICAgcmV0dXJuIEJ1ZmZlci5ieXRlTGVuZ3RoKHN0ciwgJ3V0ZjgnKTtcbiAgfVxuICByZXR1cm4gbmV3IEJsb2IoW3N0cl0pLnNpemU7XG59XG5cbi8qKlxuICogRW5jb2RlIGFuIEFycmF5QnVmZmVyIGFzIGEgYmFzZTY0IGVuY29kZWQgc3RyaW5nLlxuICpcbiAqIEBwYXJhbSBidWZmZXIgYEFycmF5QnVmZmVyYCB0byBiZSBjb252ZXJ0ZWQuXG4gKiBAcmV0dXJucyBBIHN0cmluZyB0aGF0IGJhc2U2NC1lbmNvZGVzIGBidWZmZXJgLlxuICovXG5leHBvcnQgZnVuY3Rpb24gYXJyYXlCdWZmZXJUb0Jhc2U2NFN0cmluZyhidWZmZXI6IEFycmF5QnVmZmVyKTogc3RyaW5nIHtcbiAgaWYgKHVzZU5vZGVCdWZmZXIpIHtcbiAgICByZXR1cm4gQnVmZmVyLmZyb20oYnVmZmVyKS50b1N0cmluZygnYmFzZTY0Jyk7XG4gIH1cbiAgY29uc3QgYnVmID0gbmV3IFVpbnQ4QXJyYXkoYnVmZmVyKTtcbiAgbGV0IHMgPSAnJztcbiAgZm9yIChsZXQgaSA9IDAsIGwgPSBidWYubGVuZ3RoOyBpIDwgbDsgaSsrKSB7XG4gICAgcyArPSBTdHJpbmcuZnJvbUNoYXJDb2RlKGJ1ZltpXSk7XG4gIH1cbiAgcmV0dXJuIGJ0b2Eocyk7XG59XG5cbi8qKlxuICogRGVjb2RlIGEgYmFzZTY0IHN0cmluZyBhcyBhbiBBcnJheUJ1ZmZlci5cbiAqXG4gKiBAcGFyYW0gc3RyIEJhc2U2NCBzdHJpbmcuXG4gKiBAcmV0dXJucyBEZWNvZGVkIGBBcnJheUJ1ZmZlcmAuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBiYXNlNjRTdHJpbmdUb0FycmF5QnVmZmVyKHN0cjogc3RyaW5nKTogQXJyYXlCdWZmZXIge1xuICBpZiAodXNlTm9kZUJ1ZmZlcikge1xuICAgIGNvbnN0IGJ1ZiA9IEJ1ZmZlci5mcm9tKHN0ciwgJ2Jhc2U2NCcpO1xuICAgIHJldHVybiBidWYuYnVmZmVyLnNsaWNlKGJ1Zi5ieXRlT2Zmc2V0LCBidWYuYnl0ZU9mZnNldCArIGJ1Zi5ieXRlTGVuZ3RoKTtcbiAgfVxuICBjb25zdCBzID0gYXRvYihzdHIpO1xuICBjb25zdCBidWZmZXIgPSBuZXcgVWludDhBcnJheShzLmxlbmd0aCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcy5sZW5ndGg7ICsraSkge1xuICAgIGJ1ZmZlci5zZXQoW3MuY2hhckNvZGVBdChpKV0sIGkpO1xuICB9XG4gIHJldHVybiBidWZmZXIuYnVmZmVyO1xufVxuXG4vKipcbiAqIENvbmNhdGVuYXRlIGEgbnVtYmVyIG9mIEFycmF5QnVmZmVycyBpbnRvIG9uZS5cbiAqXG4gKiBAcGFyYW0gYnVmZmVycyBBbiBhcnJheSBvZiBBcnJheUJ1ZmZlcnMgdG8gY29uY2F0ZW5hdGUsIG9yIGEgc2luZ2xlXG4gKiAgICAgQXJyYXlCdWZmZXIuXG4gKiBAcmV0dXJucyBSZXN1bHQgb2YgY29uY2F0ZW5hdGluZyBgYnVmZmVyc2AgaW4gb3JkZXIuXG4gKlxuICogQGRlcHJlY2F0ZWQgVXNlIHRmLmlvLkNvbXBvc2l0ZUFycmF5QnVmZmVyLmpvaW4oKSBpbnN0ZWFkLlxuICovXG5leHBvcnQgZnVuY3Rpb24gY29uY2F0ZW5hdGVBcnJheUJ1ZmZlcnMoYnVmZmVyczogQXJyYXlCdWZmZXJbXVxuICAgICAgfCBBcnJheUJ1ZmZlcik6IEFycmF5QnVmZmVyIHtcbiAgcmV0dXJuIENvbXBvc2l0ZUFycmF5QnVmZmVyLmpvaW4oYnVmZmVycyk7XG59XG5cbi8qKlxuICogR2V0IHRoZSBiYXNlbmFtZSBvZiBhIHBhdGguXG4gKlxuICogQmVoYXZlcyBpbiBhIHdheSBhbmFsb2dvdXMgdG8gTGludXgncyBiYXNlbmFtZSBjb21tYW5kLlxuICpcbiAqIEBwYXJhbSBwYXRoXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBiYXNlbmFtZShwYXRoOiBzdHJpbmcpOiBzdHJpbmcge1xuICBjb25zdCBTRVBBUkFUT1IgPSAnLyc7XG4gIHBhdGggPSBwYXRoLnRyaW0oKTtcbiAgd2hpbGUgKHBhdGguZW5kc1dpdGgoU0VQQVJBVE9SKSkge1xuICAgIHBhdGggPSBwYXRoLnNsaWNlKDAsIHBhdGgubGVuZ3RoIC0gMSk7XG4gIH1cbiAgY29uc3QgaXRlbXMgPSBwYXRoLnNwbGl0KFNFUEFSQVRPUik7XG4gIHJldHVybiBpdGVtc1tpdGVtcy5sZW5ndGggLSAxXTtcbn1cblxuLyoqXG4gKiBDcmVhdGUgYE1vZGVsSlNPTmAgZnJvbSBgTW9kZWxBcnRpZmFjdHNgLlxuICpcbiAqIEBwYXJhbSBhcnRpZmFjdHMgTW9kZWwgYXJ0aWZhY3RzLCBkZXNjcmliaW5nIHRoZSBtb2RlbCBhbmQgaXRzIHdlaWdodHMuXG4gKiBAcGFyYW0gbWFuaWZlc3QgV2VpZ2h0IG1hbmlmZXN0LCBkZXNjcmliaW5nIHdoZXJlIHRoZSB3ZWlnaHRzIG9mIHRoZVxuICogICAgIGBNb2RlbEFydGlmYWN0c2AgYXJlIHN0b3JlZCwgYW5kIHNvbWUgbWV0YWRhdGEgYWJvdXQgdGhlbS5cbiAqIEByZXR1cm5zIE9iamVjdCByZXByZXNlbnRpbmcgdGhlIGBtb2RlbC5qc29uYCBmaWxlIGRlc2NyaWJpbmcgdGhlIG1vZGVsXG4gKiAgICAgYXJ0aWZhY3RzIGFuZCB3ZWlnaHRzXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnZXRNb2RlbEpTT05Gb3JNb2RlbEFydGlmYWN0cyhcbiAgICBhcnRpZmFjdHM6IE1vZGVsQXJ0aWZhY3RzLCBtYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnKTogTW9kZWxKU09OIHtcbiAgY29uc3QgcmVzdWx0OiBNb2RlbEpTT04gPSB7XG4gICAgbW9kZWxUb3BvbG9neTogYXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3ksXG4gICAgZm9ybWF0OiBhcnRpZmFjdHMuZm9ybWF0LFxuICAgIGdlbmVyYXRlZEJ5OiBhcnRpZmFjdHMuZ2VuZXJhdGVkQnksXG4gICAgY29udmVydGVkQnk6IGFydGlmYWN0cy5jb252ZXJ0ZWRCeSxcbiAgICB3ZWlnaHRzTWFuaWZlc3Q6IG1hbmlmZXN0XG4gIH07XG4gIGlmIChhcnRpZmFjdHMuc2lnbmF0dXJlICE9IG51bGwpIHtcbiAgICByZXN1bHQuc2lnbmF0dXJlID0gYXJ0aWZhY3RzLnNpZ25hdHVyZTtcbiAgfVxuICBpZiAoYXJ0aWZhY3RzLnVzZXJEZWZpbmVkTWV0YWRhdGEgIT0gbnVsbCkge1xuICAgIHJlc3VsdC51c2VyRGVmaW5lZE1ldGFkYXRhID0gYXJ0aWZhY3RzLnVzZXJEZWZpbmVkTWV0YWRhdGE7XG4gIH1cbiAgaWYgKGFydGlmYWN0cy5tb2RlbEluaXRpYWxpemVyICE9IG51bGwpIHtcbiAgICByZXN1bHQubW9kZWxJbml0aWFsaXplciA9IGFydGlmYWN0cy5tb2RlbEluaXRpYWxpemVyO1xuICB9XG4gIGlmIChhcnRpZmFjdHMuaW5pdGlhbGl6ZXJTaWduYXR1cmUgIT0gbnVsbCkge1xuICAgIHJlc3VsdC5pbml0aWFsaXplclNpZ25hdHVyZSA9IGFydGlmYWN0cy5pbml0aWFsaXplclNpZ25hdHVyZTtcbiAgfVxuICBpZiAoYXJ0aWZhY3RzLnRyYWluaW5nQ29uZmlnICE9IG51bGwpIHtcbiAgICByZXN1bHQudHJhaW5pbmdDb25maWcgPSBhcnRpZmFjdHMudHJhaW5pbmdDb25maWc7XG4gIH1cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuLyoqXG4gKiBDcmVhdGUgYE1vZGVsQXJ0aWZhY3RzYCBmcm9tIGEgSlNPTiBmaWxlIGFuZCB3ZWlnaHRzLlxuICpcbiAqIEBwYXJhbSBtb2RlbEpTT04gT2JqZWN0IGNvbnRhaW5pbmcgdGhlIHBhcnNlZCBKU09OIG9mIGBtb2RlbC5qc29uYFxuICogQHBhcmFtIHdlaWdodFNwZWNzIFRoZSBsaXN0IG9mIFdlaWdodHNNYW5pZmVzdEVudHJ5IGZvciB0aGUgbW9kZWwuIE11c3QgYmVcbiAqICAgICBwYXNzZWQgaWYgdGhlIG1vZGVsSlNPTiBoYXMgYSB3ZWlnaHRzTWFuaWZlc3QuXG4gKiBAcGFyYW0gd2VpZ2h0RGF0YSBBbiBBcnJheUJ1ZmZlciBvciBhcnJheSBvZiBBcnJheUJ1ZmZlcnMgb2Ygd2VpZ2h0IGRhdGEgZm9yXG4gKiAgICAgdGhlIG1vZGVsIGNvcnJlc3BvbmRpbmcgdG8gdGhlIHdlaWdodHMgaW4gd2VpZ2h0U3BlY3MuIE11c3QgYmUgcGFzc2VkIGlmXG4gKiAgICAgdGhlIG1vZGVsSlNPTiBoYXMgYSB3ZWlnaHRzTWFuaWZlc3QuXG4gKiBAcmV0dXJucyBBIFByb21pc2Ugb2YgdGhlIGBNb2RlbEFydGlmYWN0c2AsIGFzIGRlc2NyaWJlZCBieSB0aGUgSlNPTiBmaWxlLlxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2V0TW9kZWxBcnRpZmFjdHNGb3JKU09OU3luYyhcbiAgICBtb2RlbEpTT046IE1vZGVsSlNPTiwgd2VpZ2h0U3BlY3M/OiBXZWlnaHRzTWFuaWZlc3RFbnRyeVtdLFxuICAgIHdlaWdodERhdGE/OiBXZWlnaHREYXRhKTogTW9kZWxBcnRpZmFjdHMge1xuXG4gIGNvbnN0IG1vZGVsQXJ0aWZhY3RzOiBNb2RlbEFydGlmYWN0cyA9IHtcbiAgICBtb2RlbFRvcG9sb2d5OiBtb2RlbEpTT04ubW9kZWxUb3BvbG9neSxcbiAgICBmb3JtYXQ6IG1vZGVsSlNPTi5mb3JtYXQsXG4gICAgZ2VuZXJhdGVkQnk6IG1vZGVsSlNPTi5nZW5lcmF0ZWRCeSxcbiAgICBjb252ZXJ0ZWRCeTogbW9kZWxKU09OLmNvbnZlcnRlZEJ5XG4gIH07XG5cbiAgaWYgKG1vZGVsSlNPTi50cmFpbmluZ0NvbmZpZyAhPSBudWxsKSB7XG4gICAgbW9kZWxBcnRpZmFjdHMudHJhaW5pbmdDb25maWcgPSBtb2RlbEpTT04udHJhaW5pbmdDb25maWc7XG4gIH1cbiAgaWYgKG1vZGVsSlNPTi53ZWlnaHRzTWFuaWZlc3QgIT0gbnVsbCkge1xuICAgIGlmICghd2VpZ2h0U3BlY3MpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignbW9kZWxKU09OIGhhcyB3ZWlnaHRzTWFuaWZlc3QgYnV0IHdlaWdodFNwZWNzIGlzIG51bGwnKTtcbiAgICB9XG4gICAgaWYgKCF3ZWlnaHREYXRhKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ21vZGVsSlNPTiBoYXMgd2VpZ2h0c01hbmlmZXN0IGJ1dCB3ZWlnaHREYXRhIGlzIG51bGwnKTtcbiAgICB9XG4gICAgbW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MgPSB3ZWlnaHRTcGVjcztcbiAgICBtb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhID0gd2VpZ2h0RGF0YTtcbiAgfVxuICBpZiAobW9kZWxKU09OLnNpZ25hdHVyZSAhPSBudWxsKSB7XG4gICAgbW9kZWxBcnRpZmFjdHMuc2lnbmF0dXJlID0gbW9kZWxKU09OLnNpZ25hdHVyZTtcbiAgfVxuICBpZiAobW9kZWxKU09OLnVzZXJEZWZpbmVkTWV0YWRhdGEgIT0gbnVsbCkge1xuICAgIG1vZGVsQXJ0aWZhY3RzLnVzZXJEZWZpbmVkTWV0YWRhdGEgPSBtb2RlbEpTT04udXNlckRlZmluZWRNZXRhZGF0YTtcbiAgfVxuICBpZiAobW9kZWxKU09OLm1vZGVsSW5pdGlhbGl6ZXIgIT0gbnVsbCkge1xuICAgIG1vZGVsQXJ0aWZhY3RzLm1vZGVsSW5pdGlhbGl6ZXIgPSBtb2RlbEpTT04ubW9kZWxJbml0aWFsaXplcjtcbiAgfVxuICBpZiAobW9kZWxKU09OLmluaXRpYWxpemVyU2lnbmF0dXJlICE9IG51bGwpIHtcbiAgICBtb2RlbEFydGlmYWN0cy5pbml0aWFsaXplclNpZ25hdHVyZSA9IG1vZGVsSlNPTi5pbml0aWFsaXplclNpZ25hdHVyZTtcbiAgfVxuXG4gIHJldHVybiBtb2RlbEFydGlmYWN0cztcbn1cblxuLyoqXG4gKiBDcmVhdGUgYE1vZGVsQXJ0aWZhY3RzYCBmcm9tIGEgSlNPTiBmaWxlLlxuICpcbiAqIEBwYXJhbSBtb2RlbEpTT04gT2JqZWN0IGNvbnRhaW5pbmcgdGhlIHBhcnNlZCBKU09OIG9mIGBtb2RlbC5qc29uYFxuICogQHBhcmFtIGxvYWRXZWlnaHRzIEZ1bmN0aW9uIHRoYXQgdGFrZXMgdGhlIEpTT04gZmlsZSdzIHdlaWdodHMgbWFuaWZlc3QsXG4gKiAgICAgcmVhZHMgd2VpZ2h0cyBmcm9tIHRoZSBsaXN0ZWQgcGF0aChzKSwgYW5kIHJldHVybnMgYSBQcm9taXNlIG9mIHRoZVxuICogICAgIHdlaWdodCBtYW5pZmVzdCBlbnRyaWVzIGFsb25nIHdpdGggdGhlIHdlaWdodHMgZGF0YS5cbiAqIEByZXR1cm5zIEEgUHJvbWlzZSBvZiB0aGUgYE1vZGVsQXJ0aWZhY3RzYCwgYXMgZGVzY3JpYmVkIGJ5IHRoZSBKU09OIGZpbGUuXG4gKi9cbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBnZXRNb2RlbEFydGlmYWN0c0ZvckpTT04oXG4gICAgbW9kZWxKU09OOiBNb2RlbEpTT04sXG4gICAgbG9hZFdlaWdodHM6ICh3ZWlnaHRzTWFuaWZlc3Q6IFdlaWdodHNNYW5pZmVzdENvbmZpZykgPT4gUHJvbWlzZTxbXG4gICAgICAvKiB3ZWlnaHRTcGVjcyAqLyBXZWlnaHRzTWFuaWZlc3RFbnRyeVtdLCBXZWlnaHREYXRhLFxuICAgIF0+KTogUHJvbWlzZTxNb2RlbEFydGlmYWN0cz4ge1xuICBsZXQgd2VpZ2h0U3BlY3M6IFdlaWdodHNNYW5pZmVzdEVudHJ5W10gfCB1bmRlZmluZWQ7XG4gIGxldCB3ZWlnaHREYXRhOiBXZWlnaHREYXRhIHwgdW5kZWZpbmVkO1xuXG4gIGlmIChtb2RlbEpTT04ud2VpZ2h0c01hbmlmZXN0ICE9IG51bGwpIHtcbiAgICBbd2VpZ2h0U3BlY3MsIHdlaWdodERhdGFdID0gYXdhaXQgbG9hZFdlaWdodHMobW9kZWxKU09OLndlaWdodHNNYW5pZmVzdCk7XG4gIH1cblxuICByZXR1cm4gZ2V0TW9kZWxBcnRpZmFjdHNGb3JKU09OU3luYyhtb2RlbEpTT04sIHdlaWdodFNwZWNzLCB3ZWlnaHREYXRhKTtcbn1cblxuLyoqXG4gKiBQb3B1bGF0ZSBNb2RlbEFydGlmYWN0c0luZm8gZmllbGRzIGZvciBhIG1vZGVsIHdpdGggSlNPTiB0b3BvbG9neS5cbiAqIEBwYXJhbSBtb2RlbEFydGlmYWN0c1xuICogQHJldHVybnMgQSBNb2RlbEFydGlmYWN0c0luZm8gb2JqZWN0LlxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2V0TW9kZWxBcnRpZmFjdHNJbmZvRm9ySlNPTihtb2RlbEFydGlmYWN0czogTW9kZWxBcnRpZmFjdHMpOlxuICAgIE1vZGVsQXJ0aWZhY3RzSW5mbyB7XG4gIGlmIChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5IGluc3RhbmNlb2YgQXJyYXlCdWZmZXIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0V4cGVjdGVkIEpTT04gbW9kZWwgdG9wb2xvZ3ksIHJlY2VpdmVkIEFycmF5QnVmZmVyLicpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBkYXRlU2F2ZWQ6IG5ldyBEYXRlKCksXG4gICAgbW9kZWxUb3BvbG9neVR5cGU6ICdKU09OJyxcbiAgICBtb2RlbFRvcG9sb2d5Qnl0ZXM6IG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kgPT0gbnVsbCA/XG4gICAgICAgIDAgOlxuICAgICAgICBzdHJpbmdCeXRlTGVuZ3RoKEpTT04uc3RyaW5naWZ5KG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kpKSxcbiAgICB3ZWlnaHRTcGVjc0J5dGVzOiBtb2RlbEFydGlmYWN0cy53ZWlnaHRTcGVjcyA9PSBudWxsID9cbiAgICAgICAgMCA6XG4gICAgICAgIHN0cmluZ0J5dGVMZW5ndGgoSlNPTi5zdHJpbmdpZnkobW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MpKSxcbiAgICB3ZWlnaHREYXRhQnl0ZXM6IG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEgPT0gbnVsbCA/XG4gICAgICAgIDAgOlxuICAgICAgICBuZXcgQ29tcG9zaXRlQXJyYXlCdWZmZXIobW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YSkuYnl0ZUxlbmd0aCxcbiAgfTtcbn1cblxuLyoqXG4gKiBDb25jYXRlbmF0ZSB0aGUgd2VpZ2h0cyBzdG9yZWQgaW4gYSBXZWlnaHRzTWFuaWZlc3RDb25maWcgaW50byBhIGxpc3Qgb2ZcbiAqIFdlaWdodHNNYW5pZmVzdEVudHJ5XG4gKlxuICogQHBhcmFtIHdlaWdodHNNYW5pZmVzdCBUaGUgV2VpZ2h0c01hbmlmZXN0Q29uZmlnIHRvIGV4dHJhY3Qgd2VpZ2h0cyBmcm9tLlxuICogQHJldHVybnMgQSBsaXN0IG9mIFdlaWdodHNNYW5pZmVzdEVudHJ5IG9mIHRoZSB3ZWlnaHRzIGluIHRoZSB3ZWlnaHRzTWFuaWZlc3RcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdldFdlaWdodFNwZWNzKHdlaWdodHNNYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnKTpcbiAgICBXZWlnaHRzTWFuaWZlc3RFbnRyeVtdIHtcbiAgY29uc3Qgd2VpZ2h0U3BlY3M6IFdlaWdodHNNYW5pZmVzdEVudHJ5W10gPSBbXTtcbiAgZm9yIChjb25zdCBlbnRyeSBvZiB3ZWlnaHRzTWFuaWZlc3QpIHtcbiAgICB3ZWlnaHRTcGVjcy5wdXNoKC4uLmVudHJ5LndlaWdodHMpO1xuICB9XG4gIHJldHVybiB3ZWlnaHRTcGVjcztcbn1cblxuLyoqXG4gKiBDb21wdXRlcyBtYW50aXNhIHRhYmxlIGZvciBjYXN0aW5nIEZsb2F0MTYgdG8gRmxvYXQzMlxuICogU2VlIGh0dHA6Ly93d3cuZm94LXRvb2xraXQub3JnL2Z0cC9mYXN0aGFsZmZsb2F0Y29udmVyc2lvbi5wZGZcbiAqXG4gKiBAcmV0dXJucyBVaW50MzJBcnJheSwgMjA0OCBtYW50aXNzYSBsb29rdXAgdmFsdWVzLlxuICovXG5mdW5jdGlvbiBjb21wdXRlRmxvYXQxNk1hbnRpc2FUYWJsZSgpOiBVaW50MzJBcnJheSB7XG4gIGNvbnN0IGNvbnZlcnRNYW50aXNzYSA9IChpOiBudW1iZXIpOiBudW1iZXIgPT4ge1xuICAgIGxldCBtID0gaSA8PCAxMztcbiAgICBsZXQgZSA9IDA7XG5cbiAgICB3aGlsZSAoKG0gJiAweDAwODAwMDAwKSA9PT0gMCkge1xuICAgICAgZSAtPSAweDAwODAwMDAwO1xuICAgICAgbSA8PD0gMTtcbiAgICB9XG4gICAgbSAmPSB+MHgwMDgwMDAwMDtcbiAgICBlICs9IDB4Mzg4MDAwMDA7XG5cbiAgICByZXR1cm4gbSB8IGU7XG4gIH07XG5cbiAgY29uc3QgbWFudGlzYVRhYmxlID0gbmV3IFVpbnQzMkFycmF5KDIwNDgpO1xuXG4gIG1hbnRpc2FUYWJsZVswXSA9IDA7XG4gIGZvciAobGV0IGkgPSAxOyBpIDwgMTAyNDsgaSsrKSB7XG4gICAgbWFudGlzYVRhYmxlW2ldID0gY29udmVydE1hbnRpc3NhKGkpO1xuICB9XG4gIGZvciAobGV0IGkgPSAxMDI0OyBpIDwgMjA0ODsgaSsrKSB7XG4gICAgbWFudGlzYVRhYmxlW2ldID0gMHgzODAwMDAwMCArICgoaSAtIDEwMjQpIDw8IDEzKTtcbiAgfVxuXG4gIHJldHVybiBtYW50aXNhVGFibGU7XG59XG5cbi8qKlxuICogQ29tcHV0ZXMgZXhwb25lbnQgdGFibGUgZm9yIGNhc3RpbmcgRmxvYXQxNiB0byBGbG9hdDMyXG4gKiBTZWUgaHR0cDovL3d3dy5mb3gtdG9vbGtpdC5vcmcvZnRwL2Zhc3RoYWxmZmxvYXRjb252ZXJzaW9uLnBkZlxuICpcbiAqIEByZXR1cm5zIFVpbnQzMkFycmF5LCA2NCBleHBvbmVudCBsb29rdXAgdmFsdWVzLlxuICovXG5mdW5jdGlvbiBjb21wdXRlRmxvYXQxNkV4cG9uZW50VGFibGUoKTogVWludDMyQXJyYXkge1xuICBjb25zdCBleHBvbmVudFRhYmxlID0gbmV3IFVpbnQzMkFycmF5KDY0KTtcblxuICBleHBvbmVudFRhYmxlWzBdID0gMDtcbiAgZXhwb25lbnRUYWJsZVszMV0gPSAweDQ3ODAwMDAwO1xuICBleHBvbmVudFRhYmxlWzMyXSA9IDB4ODAwMDAwMDA7XG4gIGV4cG9uZW50VGFibGVbNjNdID0gMHhjNzgwMDAwMDtcbiAgZm9yIChsZXQgaSA9IDE7IGkgPCAzMTsgaSsrKSB7XG4gICAgZXhwb25lbnRUYWJsZVtpXSA9IGkgPDwgMjM7XG4gIH1cbiAgZm9yIChsZXQgaSA9IDMzOyBpIDwgNjM7IGkrKykge1xuICAgIGV4cG9uZW50VGFibGVbaV0gPSAweDgwMDAwMDAwICsgKChpIC0gMzIpIDw8IDIzKTtcbiAgfVxuXG4gIHJldHVybiBleHBvbmVudFRhYmxlO1xufVxuXG4vKipcbiAqIENvbXB1dGVzIG9mZnNldCB0YWJsZSBmb3IgY2FzdGluZyBGbG9hdDE2IHRvIEZsb2F0MzJcbiAqIFNlZSBodHRwOi8vd3d3LmZveC10b29sa2l0Lm9yZy9mdHAvZmFzdGhhbGZmbG9hdGNvbnZlcnNpb24ucGRmXG4gKlxuICogQHJldHVybnMgVWludDMyQXJyYXksIDZkIG9mZnNldCB2YWx1ZXMuXG4gKi9cbmZ1bmN0aW9uIGNvbXB1dGVGbG9hdDE2T2Zmc2V0VGFibGUoKTogVWludDMyQXJyYXkge1xuICBjb25zdCBvZmZzZXRUYWJsZSA9IG5ldyBVaW50MzJBcnJheSg2NCk7XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCA2NDsgaSsrKSB7XG4gICAgb2Zmc2V0VGFibGVbaV0gPSAxMDI0O1xuICB9XG4gIG9mZnNldFRhYmxlWzBdID0gb2Zmc2V0VGFibGVbMzJdID0gMDtcblxuICByZXR1cm4gb2Zmc2V0VGFibGU7XG59XG5cbi8qKlxuICogUmV0cmlldmUgYSBGbG9hdDE2IGRlY29kZXIgd2hpY2ggd2lsbCBkZWNvZGUgYSBCeXRlQXJyYXkgb2YgRmxvYXQxNiB2YWx1ZXNcbiAqIHRvIGEgRmxvYXQzMkFycmF5LlxuICpcbiAqIEByZXR1cm5zIEZ1bmN0aW9uIChidWZmZXI6IFVpbnQxNkFycmF5KSA9PiBGbG9hdDMyQXJyYXkgd2hpY2ggZGVjb2Rlc1xuICogICAgICAgICAgdGhlIFVpbnQxNkFycmF5IG9mIEZsb2F0MTYgYnl0ZXMgdG8gYSBGbG9hdDMyQXJyYXkuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnZXRGbG9hdDE2RGVjb2RlcigpOiAoYnVmZmVyOiBVaW50MTZBcnJheSkgPT4gRmxvYXQzMkFycmF5IHtcbiAgLy8gQWxnb3JpdGhtIGlzIGJhc2VkIG9mZiBvZlxuICAvLyBodHRwOi8vd3d3LmZveC10b29sa2l0Lm9yZy9mdHAvZmFzdGhhbGZmbG9hdGNvbnZlcnNpb24ucGRmXG5cbiAgLy8gQ2FjaGUgbG9va3VwIHRhYmxlc1xuICBjb25zdCBtYW50aXNhVGFibGUgPSBjb21wdXRlRmxvYXQxNk1hbnRpc2FUYWJsZSgpO1xuICBjb25zdCBleHBvbmVudFRhYmxlID0gY29tcHV0ZUZsb2F0MTZFeHBvbmVudFRhYmxlKCk7XG4gIGNvbnN0IG9mZnNldFRhYmxlID0gY29tcHV0ZUZsb2F0MTZPZmZzZXRUYWJsZSgpO1xuXG4gIHJldHVybiAocXVhbnRpemVkQXJyYXk6IFVpbnQxNkFycmF5KSA9PiB7XG4gICAgY29uc3QgYnVmZmVyID0gbmV3IEFycmF5QnVmZmVyKDQgKiBxdWFudGl6ZWRBcnJheS5sZW5ndGgpO1xuICAgIGNvbnN0IGJ1ZmZlclVpbnQzMlZpZXcgPSBuZXcgVWludDMyQXJyYXkoYnVmZmVyKTtcbiAgICBmb3IgKGxldCBpbmRleCA9IDA7IGluZGV4IDwgcXVhbnRpemVkQXJyYXkubGVuZ3RoOyBpbmRleCsrKSB7XG4gICAgICBjb25zdCBmbG9hdDE2Qml0cyA9IHF1YW50aXplZEFycmF5W2luZGV4XTtcbiAgICAgIGNvbnN0IGZsb2F0MzJCaXRzID1cbiAgICAgICAgICBtYW50aXNhVGFibGVbb2Zmc2V0VGFibGVbZmxvYXQxNkJpdHMgPj4gMTBdICsgKGZsb2F0MTZCaXRzICYgMHgzZmYpXSArXG4gICAgICAgICAgZXhwb25lbnRUYWJsZVtmbG9hdDE2Qml0cyA+PiAxMF07XG4gICAgICBidWZmZXJVaW50MzJWaWV3W2luZGV4XSA9IGZsb2F0MzJCaXRzO1xuICAgIH1cbiAgICByZXR1cm4gbmV3IEZsb2F0MzJBcnJheShidWZmZXIpO1xuICB9O1xufVxuIl19