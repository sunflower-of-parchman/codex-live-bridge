#!/usr/bin/env node
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const DEVICE_NAME = "LiveUdpBridge.amxd";
const ROUTER_NAME = "live_udp_bridge.js";
const RECEIVER_NAME = "osc_loopback_receiver.js";
const PACKAGE_NAMES = [DEVICE_NAME, ROUTER_NAME, RECEIVER_NAME];
const INSTALL_ORDER = [RECEIVER_NAME, ROUTER_NAME, DEVICE_NAME];
const TOKEN_PREFIX = "set_auth_token ";
const TOKEN_PLACEHOLDER = "CHANGE_ME_BEFORE_USE";
const PROJECT_ROOT = fs.realpathSync(path.resolve(__dirname, ".."));
const SOURCE_DIRECTORY = path.join(PROJECT_ROOT, "bridge", "m4l");
const PYTHON_BRIDGE = path.join(PROJECT_ROOT, "bridge", "ableton_udp_bridge.py");
const DEFAULT_DEVICE = path.join(
  os.homedir(),
  "Music",
  "Ableton",
  "User Library",
  "Presets",
  "MIDI Effects",
  "Max MIDI Effect",
  DEVICE_NAME
);
const DEFAULT_BACKUP_ROOT = path.join(
  os.homedir(),
  "Library",
  "Application Support",
  "codex-live-bridge",
  "backups"
);

const HELP = `Usage: node scripts/ableton-device.js [options]

Rebuild the local Ableton bridge from its existing MIDI Max for Live device.
Without --install, only a private staged package is created.

Options:
  --device PATH       Existing LiveUdpBridge.amxd to use as the baseline.
  --output-dir DIR    Private directory for the three staged bridge files.
  --backup-dir DIR    Root for private timestamped installation backups.
  --install           Explicitly install the verified staged bridge package.
  --verify-live       Verify read-only Live status after installation.
  --python PATH       Python executable used by --verify-live.
  --help              Show this help message.
`;

function fail(message) {
  throw new Error(message);
}

function parseArguments(argv) {
  const options = {
    device: DEFAULT_DEVICE,
    outputDirectory: null,
    backupRoot: DEFAULT_BACKUP_ROOT,
    install: false,
    verifyLive: false,
    python: "python3",
    help: false,
  };
  const valueOptions = new Map([
    ["--device", "device"],
    ["--output-dir", "outputDirectory"],
    ["--backup-dir", "backupRoot"],
    ["--python", "python"],
  ]);
  const seen = new Set();

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") {
      options.help = true;
      continue;
    }
    if (seen.has(argument)) {
      fail(`Option may only be provided once: ${argument}`);
    }
    seen.add(argument);

    if (valueOptions.has(argument)) {
      const value = argv[index + 1];
      if (!value || value.startsWith("--") || value.includes("\0")) {
        fail(`Option requires a nonempty path: ${argument}`);
      }
      options[valueOptions.get(argument)] = value;
      index += 1;
      continue;
    }
    if (argument === "--install") {
      options.install = true;
      continue;
    }
    if (argument === "--verify-live") {
      options.verifyLive = true;
      continue;
    }
    fail(`Unknown option: ${argument}`);
  }

  if (options.verifyLive && !options.install) {
    fail("--verify-live requires --install");
  }
  return options;
}

function canonicalCandidate(target) {
  let existing = path.resolve(target);
  const remainder = [];
  while (!fs.existsSync(existing)) {
    const parent = path.dirname(existing);
    if (parent === existing) {
      fail("Could not resolve a safe existing parent directory");
    }
    remainder.unshift(path.basename(existing));
    existing = parent;
  }
  return path.resolve(fs.realpathSync(existing), ...remainder);
}

function isWithin(candidate, parent) {
  const relative = path.relative(parent, candidate);
  return relative === "" || (!relative.startsWith(".." + path.sep) && relative !== "..");
}

function rejectRepositoryArtifact(target, label) {
  const canonical = canonicalCandidate(target);
  if (isWithin(canonical, PROJECT_ROOT)) {
    fail(`${label} must be outside the repository because it may contain a private token`);
  }
  return canonical;
}

function existingStat(target, label) {
  let stat;
  try {
    stat = fs.lstatSync(target);
  } catch (error) {
    if (error && error.code === "ENOENT") {
      return null;
    }
    fail(`Could not inspect ${label}`);
  }
  if (stat.isSymbolicLink()) {
    fail(`${label} must not be a symbolic link`);
  }
  return stat;
}

function requireRegularFile(target, label) {
  const stat = existingStat(target, label);
  if (!stat || !stat.isFile()) {
    fail(`${label} must be an existing regular file`);
  }
  return stat;
}

function ensurePrivateDirectory(target, label) {
  const resolved = path.resolve(target);
  const previous = existingStat(resolved, label);
  if (previous && !previous.isDirectory()) {
    fail(`${label} must be a directory`);
  }
  if (!previous) {
    fs.mkdirSync(resolved, { recursive: true, mode: 0o700 });
  }
  fs.chmodSync(resolved, 0o700);
  const canonical = fs.realpathSync(resolved);
  if (isWithin(canonical, PROJECT_ROOT)) {
    fail(`${label} must be outside the repository because it may contain a private token`);
  }
  return canonical;
}

function temporarySibling(destination, label) {
  return path.join(
    path.dirname(destination),
    `.${path.basename(destination)}.${label}-${process.pid}-${crypto.randomBytes(8).toString("hex")}`
  );
}

function removeTemporary(target) {
  try {
    fs.unlinkSync(target);
  } catch (error) {
    if (!error || error.code !== "ENOENT") {
      throw error;
    }
  }
}

function assertReplaceable(target, label) {
  const stat = existingStat(target, label);
  if (stat && !stat.isFile()) {
    fail(`${label} must be a regular file`);
  }
}

function writeBufferAtomically(destination, content, mode = 0o600) {
  assertReplaceable(destination, "Staged bridge file");
  const temporary = temporarySibling(destination, "codex-stage");
  let descriptor;
  try {
    descriptor = fs.openSync(temporary, "wx", 0o600);
    fs.writeFileSync(descriptor, content);
    fs.fchmodSync(descriptor, mode);
    fs.closeSync(descriptor);
    descriptor = undefined;
    fs.renameSync(temporary, destination);
  } catch (error) {
    if (descriptor !== undefined) {
      fs.closeSync(descriptor);
    }
    removeTemporary(temporary);
    throw error;
  }
}

function copyFileAtomically(source, destination, mode = 0o600) {
  requireRegularFile(source, "Bridge package source file");
  assertReplaceable(destination, "Bridge package destination file");
  const temporary = temporarySibling(destination, "codex-install");
  try {
    fs.copyFileSync(source, temporary, fs.constants.COPYFILE_EXCL);
    fs.chmodSync(temporary, mode);
    fs.renameSync(temporary, destination);
  } catch (error) {
    removeTemporary(temporary);
    throw error;
  }
}

function parseDevice(devicePath, label) {
  requireRegularFile(devicePath, label);
  const data = fs.readFileSync(devicePath);
  if (
    data.length < 33 ||
    data.toString("ascii", 0, 4) !== "ampf" ||
    data.readUInt32LE(4) !== 4 ||
    data.toString("ascii", 8, 12) !== "mmmm" ||
    data.toString("ascii", 12, 16) !== "meta" ||
    data.readUInt32LE(16) !== 4 ||
    data.toString("ascii", 24, 28) !== "ptch"
  ) {
    fail(`${label} is not a recognized MIDI Max for Live device`);
  }
  const payloadLength = data.readUInt32LE(28);
  if (payloadLength !== data.length - 32 || data[data.length - 1] !== 0) {
    fail(`${label} has an unsupported payload length or trailing terminator`);
  }

  let document;
  try {
    document = JSON.parse(data.subarray(32, -1).toString("utf8"));
  } catch (_error) {
    fail(`${label} contains invalid Max patch JSON`);
  }
  if (!document || !document.patcher || !Array.isArray(document.patcher.boxes)) {
    fail(`${label} does not contain a valid Max patch`);
  }
  return { data, document };
}

function parseSourcePatch(sourcePath) {
  requireRegularFile(sourcePath, "Repository Max patch");
  let document;
  try {
    document = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
  } catch (_error) {
    fail("Repository Max patch contains invalid JSON");
  }
  if (!document || !document.patcher || !Array.isArray(document.patcher.boxes)) {
    fail("Repository Max patch does not contain a valid patcher");
  }
  return document;
}

function authBox(document, label) {
  const matches = document.patcher.boxes.filter(({ box }) =>
    typeof box?.text === "string" && box.text.startsWith(TOKEN_PREFIX)
  );
  if (matches.length !== 1) {
    fail(`${label} must contain exactly one local capability-token setup box`);
  }
  return matches[0].box;
}

function validateSecurePatch(document, expectedTokenText) {
  const boxes = document.patcher.boxes.map(({ box }) => box || {});
  const receivers = boxes.filter((box) =>
    typeof box.text === "string" && box.text.startsWith("node.script osc_loopback_receiver.js")
  );
  if (receivers.length !== 1) {
    fail("Rebuilt device must contain exactly one secure loopback receiver");
  }
  const receiver = receivers[0].text;
  for (const attribute of ["@autostart 1", "@defer 1", "@restart 1"]) {
    if (!receiver.includes(attribute)) {
      fail("Rebuilt loopback receiver is missing a required safety setting");
    }
  }
  if (boxes.some((box) => String(box.text || "").startsWith("udpreceive"))) {
    fail("Rebuilt device must not contain an unrestricted udpreceive object");
  }
  const routers = boxes.filter((box) => box.text === "js live_udp_bridge.js");
  if (routers.length !== 1 || Number(routers[0].numinlets) !== 2) {
    fail("Rebuilt device must preserve isolated local capability-token setup");
  }
  const dependencies = new Set(
    (document.patcher.dependency_cache || []).map((dependency) => dependency?.name)
  );
  if (!dependencies.has(ROUTER_NAME) || !dependencies.has(RECEIVER_NAME)) {
    fail("Rebuilt device is missing a required JavaScript dependency");
  }
  if (authBox(document, "Rebuilt device").text !== expectedTokenText) {
    fail("Rebuilt device did not preserve its existing capability token");
  }
}

function sha256(target) {
  return crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex");
}

function rebuildPackage(installed, stageDirectory) {
  const source = parseSourcePatch(path.join(SOURCE_DIRECTORY, "LiveUdpBridge.maxpat"));
  const installedTokenBox = authBox(installed.document, "Existing Ableton device");
  const tokenText = installedTokenBox.text;
  const sourceTokenBox = authBox(source, "Repository Max patch");
  sourceTokenBox.text = tokenText;

  const document = {
    ...installed.document,
    ...source,
    patcher: {
      ...installed.document.patcher,
      ...source.patcher,
      appversion: installed.document.patcher.appversion,
      project: installed.document.patcher.project,
    },
  };
  validateSecurePatch(document, tokenText);

  const payload = Buffer.concat([
    Buffer.from(JSON.stringify(document, null, 2), "utf8"),
    Buffer.from([0]),
  ]);
  if (payload.length > 0xffffffff) {
    fail("Rebuilt patch exceeds the Max for Live container size limit");
  }
  const header = Buffer.from(installed.data.subarray(0, 32));
  header.writeUInt32LE(payload.length, 28);

  const stagedDevice = path.join(stageDirectory, DEVICE_NAME);
  writeBufferAtomically(stagedDevice, Buffer.concat([header, payload]), 0o600);
  for (const name of [ROUTER_NAME, RECEIVER_NAME]) {
    copyFileAtomically(
      path.join(SOURCE_DIRECTORY, name),
      path.join(stageDirectory, name),
      0o600
    );
  }

  const rebuilt = parseDevice(stagedDevice, "Staged Ableton device");
  validateSecurePatch(rebuilt.document, tokenText);
  if (!installed.data.subarray(0, 28).equals(rebuilt.data.subarray(0, 28))) {
    fail("Rebuilt device did not preserve the original MIDI container metadata");
  }
  if (
    JSON.stringify(rebuilt.document.patcher.appversion) !==
      JSON.stringify(installed.document.patcher.appversion) ||
    JSON.stringify(rebuilt.document.patcher.project) !==
      JSON.stringify(installed.document.patcher.project)
  ) {
    fail("Rebuilt device did not preserve its existing Max version or project metadata");
  }

  const hashes = Object.fromEntries(
    PACKAGE_NAMES.map((name) => [name, sha256(path.join(stageDirectory, name))])
  );
  for (const name of [ROUTER_NAME, RECEIVER_NAME]) {
    if (hashes[name] !== sha256(path.join(SOURCE_DIRECTORY, name))) {
      fail(`Staged bridge dependency does not match repository source: ${name}`);
    }
  }
  const configuredToken = tokenText.slice(TOKEN_PREFIX.length).trim();
  return {
    hashes,
    tokenConfigured: configuredToken.length > 0 && configuredToken !== TOKEN_PLACEHOLDER,
  };
}

function inspectInstallationTargets(destinationDirectory) {
  return Object.fromEntries(
    PACKAGE_NAMES.map((name) => {
      const destination = path.join(destinationDirectory, name);
      const stat = existingStat(destination, `Installed bridge file ${name}`);
      if (stat && !stat.isFile()) {
        fail(`Installed bridge file ${name} must be a regular file`);
      }
      return [
        name,
        {
          existed: Boolean(stat),
          mode: stat ? stat.mode & 0o777 : 0o644,
        },
      ];
    })
  );
}

function createBackup(backupRoot, destinationDirectory, existing) {
  const root = path.resolve(backupRoot);
  ensurePrivateDirectory(root, "Persistent backup root");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const backupDirectory = fs.mkdtempSync(path.join(root, `${timestamp}-`));
  fs.chmodSync(backupDirectory, 0o700);

  for (const name of PACKAGE_NAMES) {
    if (existing[name].existed) {
      copyFileAtomically(
        path.join(destinationDirectory, name),
        path.join(backupDirectory, name),
        0o600
      );
    }
  }
  return backupDirectory;
}

function verifyInstalledHashes(stageDirectory, destinationDirectory, hashes) {
  for (const name of PACKAGE_NAMES) {
    const staged = path.join(stageDirectory, name);
    const installed = path.join(destinationDirectory, name);
    if (sha256(staged) !== hashes[name] || sha256(installed) !== hashes[name]) {
      fail(`Installed bridge file does not match the staged package: ${name}`);
    }
  }
}

function verifyLiveBridge(python) {
  const arguments_ = [
    PYTHON_BRIDGE,
    "--ack",
    "--status",
    "--no-tempo",
    "--no-signature",
    "--no-metrics",
    "--ack-timeout",
    "2",
  ];
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const result = spawnSync(python, arguments_, {
      cwd: PROJECT_ROOT,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 5000,
      maxBuffer: 64 * 1024,
    });
    if (!result.error && result.status === 0) {
      return;
    }
    if (attempt === 0) {
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 150);
    }
  }
  fail("Installed bridge live status verification failed; installation was rolled back");
}

function rollbackInstallation(touched, existing, backupDirectory, destinationDirectory) {
  const failures = [];
  for (const name of touched.slice().reverse()) {
    const destination = path.join(destinationDirectory, name);
    try {
      if (existing[name].existed) {
        copyFileAtomically(
          path.join(backupDirectory, name),
          destination,
          existing[name].mode
        );
      } else {
        const current = existingStat(destination, `Installed bridge file ${name}`);
        if (current) {
          fs.unlinkSync(destination);
        }
      }
    } catch (_error) {
      failures.push(name);
    }
  }
  if (failures.length > 0) {
    fail(`Automatic rollback could not restore: ${failures.join(", ")}`);
  }
}

function installPackage(stageDirectory, destinationDirectory, backupRoot, hashes, options) {
  const existing = inspectInstallationTargets(destinationDirectory);
  if (!existing[DEVICE_NAME].existed) {
    fail("The existing Ableton device disappeared before installation");
  }
  const backupDirectory = createBackup(backupRoot, destinationDirectory, existing);
  const touched = [];

  try {
    for (const name of INSTALL_ORDER) {
      const mode = name === DEVICE_NAME ? 0o600 : existing[name].mode;
      copyFileAtomically(
        path.join(stageDirectory, name),
        path.join(destinationDirectory, name),
        mode
      );
      touched.push(name);
    }
    verifyInstalledHashes(stageDirectory, destinationDirectory, hashes);
    if (options.verifyLive) {
      verifyLiveBridge(options.python);
    }
  } catch (error) {
    try {
      rollbackInstallation(touched, existing, backupDirectory, destinationDirectory);
    } catch (rollbackError) {
      fail(`${error.message}; ${rollbackError.message}`);
    }
    throw error;
  }

  return backupDirectory;
}

function resolveDevice(target) {
  const resolved = path.resolve(target);
  if (path.basename(resolved) !== DEVICE_NAME) {
    fail(`The existing Ableton device must be named ${DEVICE_NAME}`);
  }
  requireRegularFile(resolved, "Existing Ableton device");
  const destinationDirectory = fs.realpathSync(path.dirname(resolved));
  return {
    device: path.join(destinationDirectory, DEVICE_NAME),
    destinationDirectory,
  };
}

function ensureDisjointArtifactLocations(stageDirectory, backupRoot, destinationDirectory) {
  if (isWithin(stageDirectory, destinationDirectory)) {
    fail("Staging directory must not overlap the installed Ableton device directory");
  }
  if (isWithin(backupRoot, destinationDirectory)) {
    fail("Persistent backup root must not overlap the installed Ableton device directory");
  }
  if (isWithin(stageDirectory, backupRoot) || isWithin(backupRoot, stageDirectory)) {
    fail("Staging directory and persistent backup root must not overlap");
  }
}

function resolveStageDirectory(requested, destinationDirectory, backupRoot) {
  if (requested !== null) {
    const candidate = rejectRepositoryArtifact(requested, "Staging directory");
    ensureDisjointArtifactLocations(candidate, backupRoot, destinationDirectory);
    return ensurePrivateDirectory(requested, "Staging directory");
  }
  const temporaryRoot = rejectRepositoryArtifact(os.tmpdir(), "Temporary staging root");
  if (isWithin(temporaryRoot, destinationDirectory)) {
    fail("Temporary staging root must not overlap the installed Ableton device directory");
  }
  const temporary = fs.mkdtempSync(path.join(temporaryRoot, "codex-live-bridge-device-"));
  fs.chmodSync(temporary, 0o700);
  const stageDirectory = fs.realpathSync(temporary);
  try {
    ensureDisjointArtifactLocations(stageDirectory, backupRoot, destinationDirectory);
  } catch (error) {
    fs.rmdirSync(stageDirectory);
    throw error;
  }
  return stageDirectory;
}

function main(argv) {
  const options = parseArguments(argv);
  if (options.help) {
    process.stdout.write(HELP);
    return;
  }

  const { device, destinationDirectory } = resolveDevice(options.device);
  const baseline = parseDevice(device, "Existing Ableton device");
  authBox(baseline.document, "Existing Ableton device");
  const backupRoot = rejectRepositoryArtifact(
    options.backupRoot,
    "Persistent backup root"
  );
  if (isWithin(backupRoot, destinationDirectory)) {
    fail("Persistent backup root must not overlap the installed Ableton device directory");
  }
  const stageDirectory = resolveStageDirectory(
    options.outputDirectory,
    destinationDirectory,
    backupRoot
  );
  const { hashes, tokenConfigured } = rebuildPackage(baseline, stageDirectory);
  let backupDirectory = null;
  if (options.install) {
    backupDirectory = installPackage(
      stageDirectory,
      destinationDirectory,
      options.backupRoot,
      hashes,
      options
    );
  }

  process.stdout.write(
    JSON.stringify({
      stageDir: stageDirectory,
      installed: options.install,
      verifiedLive: options.verifyLive,
      backupDir: backupDirectory,
      tokenConfigured,
      hashes,
    }) + "\n"
  );
}

try {
  main(process.argv.slice(2));
} catch (error) {
  process.stderr.write(`error: ${error && error.message ? error.message : "operation failed"}\n`);
  process.exitCode = 1;
}
