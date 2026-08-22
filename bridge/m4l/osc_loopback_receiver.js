"use strict";

// Node for Max UDP receiver for the public bridge command channel.
// The explicit loopback bind is the transport security boundary: Max's
// built-in [udpreceive] object accepts a port but cannot select an interface.

const dgram = require("node:dgram");
const { TextDecoder } = require("node:util");

const LOOPBACK_HOST = "127.0.0.1";
const COMMAND_PORT = 9000;
const MAX_UDP_PACKET_BYTES = 65507;
const utf8Decoder = new TextDecoder("utf-8", { fatal: true });

function pad4(length) {
  const remainder = length % 4;
  return remainder === 0 ? 0 : 4 - remainder;
}

function readOscString(packet, start) {
  if (!Number.isInteger(start) || start < 0 || start >= packet.length) {
    throw new Error("OSC string is truncated");
  }

  const end = packet.indexOf(0, start);
  if (end < 0) {
    throw new Error("OSC string is missing its NUL terminator");
  }

  let value;
  try {
    value = utf8Decoder.decode(packet.subarray(start, end));
  } catch (_error) {
    throw new Error("OSC string is not valid UTF-8");
  }

  const contentEnd = end + 1;
  const paddedEnd = contentEnd + pad4(contentEnd);
  if (paddedEnd > packet.length) {
    throw new Error("OSC string padding is truncated");
  }
  for (let index = contentEnd; index < paddedEnd; index += 1) {
    if (packet[index] !== 0) {
      throw new Error("OSC string padding must be zero");
    }
  }

  return { value, next: paddedEnd };
}

function decodeOscMessage(packetValue) {
  const packet = Buffer.isBuffer(packetValue)
    ? packetValue
    : Buffer.from(packetValue || []);
  if (packet.length === 0) {
    throw new Error("OSC packet is empty");
  }
  if (packet.length > MAX_UDP_PACKET_BYTES) {
    throw new Error("OSC packet exceeds the UDP payload limit");
  }
  if (packet.subarray(0, 7).toString("ascii") === "#bundle") {
    throw new Error("OSC bundles are not supported");
  }

  const addressResult = readOscString(packet, 0);
  const address = addressResult.value;
  if (!address.startsWith("/")) {
    throw new Error("OSC address must start with '/'");
  }

  const typeResult = readOscString(packet, addressResult.next);
  const typeTags = typeResult.value;
  if (!typeTags.startsWith(",")) {
    throw new Error("OSC type tags must start with ','");
  }

  let index = typeResult.next;
  const args = [];
  for (const tag of typeTags.slice(1)) {
    if (tag === "i") {
      if (index + 4 > packet.length) {
        throw new Error("OSC int argument is truncated");
      }
      args.push(packet.readInt32BE(index));
      index += 4;
      continue;
    }
    if (tag === "f") {
      if (index + 4 > packet.length) {
        throw new Error("OSC float argument is truncated");
      }
      const value = packet.readFloatBE(index);
      if (!Number.isFinite(value)) {
        throw new Error("OSC float argument must be finite");
      }
      args.push(value);
      index += 4;
      continue;
    }
    if (tag === "s") {
      const stringResult = readOscString(packet, index);
      args.push(stringResult.value);
      index = stringResult.next;
      continue;
    }
    throw new Error(`Unsupported OSC type tag: ${tag}`);
  }

  if (index !== packet.length) {
    throw new Error("OSC message has trailing bytes");
  }
  return { address, args };
}

function createLoopbackReceiver(options = {}) {
  const port = options.port === undefined ? COMMAND_PORT : Number(options.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    throw new Error("UDP port must be an integer from 0 through 65535");
  }

  const onMessage =
    typeof options.onMessage === "function" ? options.onMessage : () => {};
  const onPacketError =
    typeof options.onPacketError === "function"
      ? options.onPacketError
      : () => {};
  const server = dgram.createSocket({ type: "udp4", reuseAddr: false });

  server.on("message", (packet, remote) => {
    try {
      const decoded = decodeOscMessage(packet);
      onMessage(decoded.address, decoded.args, remote);
    } catch (error) {
      onPacketError(error);
    }
  });
  server.bind({
    address: LOOPBACK_HOST,
    port,
    exclusive: true,
  });
  return server;
}

function runMaxReceiver() {
  const maxAPI = require("max-api");
  let droppedPackets = 0;
  let lastDropReportMs = 0;

  const receiver = createLoopbackReceiver({
    onMessage: (address, args) => {
      Promise.resolve(maxAPI.outlet(address, ...args)).catch((error) => {
        maxAPI.post(
          `[live-bridge] Unable to forward OSC message: ${error.message}`,
          maxAPI.POST_LEVELS.ERROR
        );
      });
    },
    onPacketError: (error) => {
      droppedPackets += 1;
      const now = Date.now();
      if (now - lastDropReportMs >= 1000) {
        maxAPI.post(
          `[live-bridge] Dropped ${droppedPackets} invalid OSC packet(s): ${error.message}`,
          maxAPI.POST_LEVELS.WARN
        );
        droppedPackets = 0;
        lastDropReportMs = now;
      }
    },
  });

  receiver.on("listening", () => {
    const address = receiver.address();
    maxAPI.post(
      `[live-bridge] Command receiver listening on udp://${address.address}:${address.port}`
    );
  });
  receiver.on("error", (error) => {
    maxAPI.post(
      `[live-bridge] Command receiver failed: ${error.message}`,
      maxAPI.POST_LEVELS.ERROR
    );
  });

  const close = () => receiver.close();
  process.once("SIGINT", close);
  process.once("SIGTERM", close);
}

module.exports = {
  COMMAND_PORT,
  LOOPBACK_HOST,
  MAX_UDP_PACKET_BYTES,
  createLoopbackReceiver,
  decodeOscMessage,
};

// Node for Max imports the configured script from its own process runner, so
// the receiver is not require.main inside the actual Ableton-hosted runtime.
if (require.main === module || process.env.SCRIPT_PATH === __filename) {
  runMaxReceiver();
}
