// Ableton Live UDP bridge logic for Max for Live.
// This file is intended to be loaded via: [js live_udp_bridge.js]

autowatch = 1;
inlets = 1;
outlets = 3; // 0 -> UDP ack/debug, 1 -> console/debug, 2 -> MIDI out

var song = null;
var initialized = false;
var apiObservers = Object.create(null);
var apiObserverCounter = 0;
var MAX_API_OBSERVERS = 32;
var MAX_OBSERVER_ID_BYTES = 128;
var bridgeAuthToken = "";
var AUTH_TOKEN_PLACEHOLDER = "CHANGE_ME_BEFORE_USE";
var MIN_AUTH_TOKEN_BYTES = 16;
var MAX_AUTH_TOKEN_BYTES = 256;
var MAX_TRACKS_PER_COMMAND = 32;
var MAX_TRACK_TARGET = 256;
var ERROR_CORRELATION_MARKER = "request_correlation";
var SESSION_CLIP_INSPECTION_SCHEMA = "codex-live-bridge.session-midi-clip-inspection";
var SESSION_CLIP_INSPECTION_SCHEMA_VERSION = 1;
var SESSION_CLIP_INSPECTION_PRODUCER_VERSION = "3.1.0";
var SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES = 4096;
var SESSION_CLIP_INSPECTION_MAX_NOTES = 4096;
var SESSION_CLIP_INSPECTION_MAX_DEVICES = 256;
var SESSION_CLIP_INSPECTION_MAX_FRAGMENTS = 1024;
var sessionClipInspectionCounter = 0;

function debug(msg) {
  var text = "[live-bridge] " + msg;
  post(text + "\n");
  outlet(1, text);
}

function getScalar(api, prop) {
  var result = api.get(prop);
  if (Array.isArray(result)) {
    // LiveAPI can return either [value] or [prop, value].
    return result[result.length - 1];
  }
  return result;
}

function hasRequestId(requestId) {
  return !(requestId === undefined || requestId === null || String(requestId).length === 0);
}

function ackWithRequest(eventName, argsArray, requestId) {
  var payload = ["ack", eventName].concat(argsArray || []);
  if (eventName === "error") {
    payload.push(ERROR_CORRELATION_MARKER);
    payload.push("req:" + (hasRequestId(requestId) ? String(requestId) : ""));
  } else if (hasRequestId(requestId)) {
    payload.push(String(requestId));
  }
  emitAck(payload);
}

function nowMs() {
  return new Date().getTime();
}

function constantTimeStringEqual(leftValue, rightValue) {
  var left = String(leftValue || "");
  var right = String(rightValue || "");
  var maxLength = Math.max(left.length, right.length);
  var difference = left.length ^ right.length;
  for (var i = 0; i < maxLength; i += 1) {
    var leftCode = i < left.length ? left.charCodeAt(i) : 0;
    var rightCode = i < right.length ? right.charCodeAt(i) : 0;
    difference |= leftCode ^ rightCode;
  }
  return difference === 0;
}

function isValidAuthToken(token) {
  var text = token === undefined || token === null ? "" : String(token).trim();
  var byteLength = utf8ByteLength(text);
  return (
    text !== AUTH_TOKEN_PLACEHOLDER &&
    byteLength >= MIN_AUTH_TOKEN_BYTES &&
    byteLength <= MAX_AUTH_TOKEN_BYTES
  );
}

function set_auth_token(token) {
  var text = token === undefined || token === null ? "" : String(token).trim();
  if (!isValidAuthToken(text)) {
    bridgeAuthToken = "";
    debug(
      "Mutation authentication is disabled. Configure a unique " +
        MIN_AUTH_TOKEN_BYTES +
        "-to-" +
        MAX_AUTH_TOKEN_BYTES +
        "-byte token in the Max patch."
    );
    return;
  }
  bridgeAuthToken = text;
  debug("Mutation authentication enabled.");
}

function requireMutationAuth(commandName, suppliedToken, requestId) {
  var command = String(commandName || "mutation");
  if (!isValidAuthToken(bridgeAuthToken)) {
    ackWithRequest("error", ["auth_not_configured", command], requestId);
    return false;
  }
  if (!constantTimeStringEqual(bridgeAuthToken, suppliedToken)) {
    ackWithRequest("error", ["unauthorized_command", command], requestId);
    return false;
  }
  return true;
}

function boundedInteger(value, minimum, maximum) {
  var parsed = Number(value);
  if (
    !isFinite(parsed) ||
    Math.floor(parsed) !== parsed ||
    parsed < minimum ||
    parsed > maximum
  ) {
    return null;
  }
  return parsed;
}

function isValidObserverId(observerId) {
  var key = observerId === undefined || observerId === null
    ? ""
    : String(observerId).trim();
  if (
    key.length === 0 ||
    utf8ByteLength(key) > MAX_OBSERVER_ID_BYTES ||
    key === "__proto__" ||
    key === "prototype" ||
    key === "constructor"
  ) {
    return false;
  }
  return /^[A-Za-z0-9][A-Za-z0-9._:-]*$/.test(key);
}

function newObserverId() {
  apiObserverCounter += 1;
  return "obs_" + nowMs() + "_" + apiObserverCounter;
}

function normalizeObserverMode(value) {
  var n = Math.floor(Number(value));
  return n === 1 ? 1 : 0;
}

function normalizeObserverCallbackArgs(rawArgs) {
  if (!rawArgs || !rawArgs.length) {
    return [];
  }
  if (rawArgs.length === 1 && Array.isArray(rawArgs[0])) {
    return rawArgs[0];
  }
  return rawArgs;
}

function hasObserverEntry(observerId) {
  return Object.prototype.hasOwnProperty.call(apiObservers, String(observerId || ""));
}

function callbackArgsValue(args) {
  if (!args || !args.length) {
    return null;
  }
  if (args.length === 1) {
    return args[0];
  }
  return args;
}

function clearObserverEntry(observerId) {
  var key = observerId === undefined || observerId === null ? "" : String(observerId).trim();
  if (key.length === 0 || !hasObserverEntry(key)) {
    return null;
  }
  var entry = apiObservers[key];
  if (!entry) {
    return null;
  }
  try {
    if (entry.api) {
      entry.api.property = "";
    }
  } catch (err) {}
  delete apiObservers[key];
  return entry;
}

function clearAllObserverEntries() {
  var keys = Object.keys(apiObservers);
  for (var i = 0; i < keys.length; i += 1) {
    clearObserverEntry(keys[i]);
  }
  return keys.length;
}

function listObserverEntries() {
  var keys = Object.keys(apiObservers).sort();
  var items = [];
  for (var i = 0; i < keys.length; i += 1) {
    var key = keys[i];
    var entry = apiObservers[key];
    if (!entry) {
      continue;
    }
    items.push({
      observer_id: key,
      requested_path: entry.requested_path,
      current_path: entry.api
        ? normalizeLiveApiPath(entry.api.path, entry.current_path || entry.requested_path)
        : normalizeLiveApiPath(entry.current_path, entry.requested_path),
      property: entry.property,
      mode: Number(entry.mode || 0),
      live_id: entry.api ? Number(entry.api.id) : Number(entry.live_id || 0),
      created_ms: Number(entry.created_ms || 0),
      event_count: Number(entry.event_count || 0),
      dropped_events: Number(entry.dropped_events || 0),
      min_interval_ms: Number(entry.min_interval_ms || 0),
    });
  }
  return items;
}

function buildObserverPayload(entry, callbackArgs) {
  if (!entry || !entry.api) {
    return null;
  }
  var args = normalizeObserverCallbackArgs(callbackArgs || []);
  entry.event_count = Number(entry.event_count || 0) + 1;
  var value = callbackArgsValue(args);
  if (value === null && entry.property) {
    try {
      value = entry.api.get(entry.property);
    } catch (err) {}
  }
  return {
    observer_id: String(entry.observer_id || ""),
    requested_path: String(entry.requested_path || ""),
    current_path: normalizeLiveApiPath(entry.api.path, entry.current_path || entry.requested_path),
    property: String(entry.property || ""),
    mode: Number(entry.mode || 0),
    live_id: Number(entry.api.id || 0),
    event_count: Number(entry.event_count || 0),
    timestamp_ms: nowMs(),
    raw_args: args,
    value: value,
  };
}

function emitObserverEvent(observerId, callbackArgs) {
  var key = String(observerId || "");
  if (!hasObserverEntry(key)) {
    return;
  }
  var entry = apiObservers[key];
  if (!entry) {
    return;
  }
  var timestamp = nowMs();
  var minInterval = Number(entry.min_interval_ms || 0);
  if (minInterval > 0 && Number(entry.last_emit_ms || 0) > 0) {
    if (timestamp - Number(entry.last_emit_ms || 0) < minInterval) {
      entry.dropped_events = Number(entry.dropped_events || 0) + 1;
      return;
    }
  }
  entry.last_emit_ms = timestamp;
  var payload = buildObserverPayload(entry, callbackArgs);
  if (!payload) {
    return;
  }
  payload.dropped_events = Number(entry.dropped_events || 0);
  ack("ack", "api_event", String(observerId), safeJsonStringify(payload, "api_event"));
}

function buildObserverCallback(observerId) {
  return function () {
    emitObserverEvent(observerId, Array.prototype.slice.call(arguments));
  };
}

function safeJsonStringify(value, contextName) {
  try {
    return JSON.stringify(value);
  } catch (err) {
    debug("JSON stringify failed in " + contextName + ": " + err);
    return JSON.stringify({ error: "json_stringify_failed", context: contextName });
  }
}

function parseJsonPayload(raw, contextName, fallbackValue, requestId) {
  if (raw === undefined || raw === null) {
    return fallbackValue;
  }
  var text = String(raw);
  if (text.length === 0) {
    return fallbackValue;
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    debug("Failed to parse JSON payload in " + contextName + ": " + err);
    ackWithRequest("error", ["api_json_parse_failed", contextName], requestId);
    return null;
  }
}

function normalizeArgsArray(argsValue) {
  if (argsValue === undefined || argsValue === null) {
    return [];
  }
  if (Array.isArray(argsValue)) {
    return argsValue;
  }
  return [argsValue];
}

function normalizeLiveApiPath(pathValue, fallbackPath) {
  var text = pathValue === undefined || pathValue === null ? "" : String(pathValue).trim();
  if (text.length >= 2) {
    var first = text.charAt(0);
    var last = text.charAt(text.length - 1);
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      text = text.slice(1, -1).trim();
    }
  }
  if (text.length > 0) {
    return text;
  }
  return fallbackPath === undefined || fallbackPath === null ? "" : String(fallbackPath).trim();
}

function resolveApiOrError(path, contextName, requestId) {
  var pathText = normalizeLiveApiPath(path, "");
  if (pathText.length === 0) {
    ackWithRequest("error", ["api_invalid_path", contextName], requestId);
    return null;
  }
  try {
    var api = new LiveAPI(null, pathText);
    var id = api ? Number(api.id) : 0;
    if (!(id > 0)) {
      ackWithRequest("error", ["api_path_not_found", pathText, id], requestId);
      return null;
    }
    return api;
  } catch (err) {
    debug("Failed to resolve LiveAPI path '" + pathText + "' in " + contextName + ": " + err);
    ackWithRequest("error", ["api_path_resolve_failed", pathText], requestId);
    return null;
  }
}

function tryResolveApi(path) {
  try {
    var api = new LiveAPI(null, normalizeLiveApiPath(path, ""));
    if (api && Number(api.id) > 0) {
      return api;
    }
  } catch (err) {}
  return null;
}

function normalizeLiveValue(value, propName) {
  if (Array.isArray(value)) {
    if (value.length === 1) {
      return value[0];
    }
    if (value.length >= 2 && String(value[0]) === String(propName)) {
      return value[value.length - 1];
    }
  }
  return value;
}

function readApiProperty(api, propName) {
  try {
    return {
      ok: true,
      value: normalizeLiveValue(api.get(propName), propName),
    };
  } catch (err) {
    return { ok: false, error: String(err) };
  }
}

function readApiPropertyBag(api, propNames) {
  var bag = {};
  var errors = {};
  for (var i = 0; i < propNames.length; i += 1) {
    var prop = propNames[i];
    var result = readApiProperty(api, prop);
    if (result.ok) {
      bag[prop] = result.value;
    } else {
      errors[prop] = "read_failed";
    }
  }
  if (Object.keys(errors).length > 0) {
    bag.errors = errors;
  }
  return bag;
}

function normalizeTrackPathReference(trackRef, defaultPath) {
  var text = trackRef === undefined || trackRef === null ? "" : String(trackRef).trim();
  if (text.length === 0 || text === "default") {
    return defaultPath || "live_set tracks 0";
  }
  if (text === "master") {
    return "live_set master_track";
  }
  var returnMatch = text.match(/^return[:\s]+(\d+)$/);
  if (returnMatch) {
    return "live_set return_tracks " + Number(returnMatch[1]);
  }
  if (/^\d+$/.test(text)) {
    return "live_set tracks " + Number(text);
  }
  return text;
}

function describeApiTarget(path, propNames) {
  var api = tryResolveApi(path);
  if (!api) {
    return { path: String(path || ""), id: 0, error: "path_not_found" };
  }
  var payload = readApiPropertyBag(api, propNames || []);
  payload.path = normalizeLiveApiPath(api.path, path);
  payload.id = Number(api.id || 0);
  return payload;
}

function describeParameterApi(parameterApi, requestedPath) {
  if (!parameterApi || !(Number(parameterApi.id) > 0)) {
    return { path: String(requestedPath || ""), id: 0, error: "path_not_found" };
  }
  var payload = readApiPropertyBag(parameterApi, [
    "name",
    "original_name",
    "value",
    "min",
    "max",
    "default_value",
    "is_quantized",
    "value_items",
    "is_enabled",
    "automation_state",
  ]);
  payload.path = normalizeLiveApiPath(parameterApi.path, requestedPath);
  payload.id = Number(parameterApi.id || 0);
  return payload;
}

function describeParameterPath(parameterPath) {
  return describeParameterApi(tryResolveApi(parameterPath), parameterPath);
}

function describeDeviceApi(deviceApi, requestedPath, includeParameters) {
  if (!deviceApi || !(Number(deviceApi.id) > 0)) {
    return { path: String(requestedPath || ""), id: 0, error: "path_not_found" };
  }
  var payload = readApiPropertyBag(deviceApi, [
    "name",
    "class_name",
    "class_display_name",
    "type",
    "is_active",
    "can_have_chains",
    "can_have_drum_pads",
    "latency_in_samples",
    "latency_in_ms",
    "can_compare_ab",
    "is_using_compare_preset_b",
  ]);
  payload.path = normalizeLiveApiPath(deviceApi.path, requestedPath);
  payload.id = Number(deviceApi.id || 0);
  var parameterCount = 0;
  try {
    parameterCount = deviceApi.getcount("parameters");
  } catch (errCount) {
    parameterCount = 0;
  }
  payload.parameter_count = parameterCount;
  if (includeParameters) {
    payload.parameters = [];
    for (var i = 0; i < parameterCount; i += 1) {
      payload.parameters.push(describeParameterPath(payload.path + " parameters " + i));
    }
  }
  return payload;
}

function describeDevicesForTrackPath(trackPath) {
  var trackApi = tryResolveApi(trackPath);
  if (!trackApi) {
    return { track_path: String(trackPath || ""), error: "track_not_found", devices: [] };
  }
  var track = readApiPropertyBag(trackApi, [
    "name",
    "has_midi_input",
    "has_audio_input",
    "has_audio_output",
    "has_midi_output",
    "is_frozen",
  ]);
  track.path = normalizeLiveApiPath(trackApi.path, trackPath);
  track.id = Number(trackApi.id || 0);
  var deviceCount = 0;
  try {
    deviceCount = trackApi.getcount("devices");
  } catch (errCount) {
    deviceCount = 0;
  }
  var devices = [];
  for (var i = 0; i < deviceCount; i += 1) {
    devices.push(describeDeviceApi(tryResolveApi(track.path + " devices " + i), track.path + " devices " + i, false));
  }
  return {
    track: track,
    device_count: deviceCount,
    devices: devices,
  };
}

function parseOptionalInsertionIndex(indexValue, errorCode, targetPath, requestId) {
  if (indexValue === undefined || indexValue === null || String(indexValue).trim().length === 0) {
    return { ok: true, has_index: false, value: null };
  }
  var numeric = Number(indexValue);
  if (!(isFinite(numeric) && numeric >= 0 && Math.floor(numeric) === numeric)) {
    ackWithRequest("error", [errorCode, targetPath, indexValue], requestId);
    return { ok: false, has_index: false, value: null };
  }
  return { ok: true, has_index: true, value: numeric };
}

function isNumericScalar(value) {
  if (typeof value === "number") {
    return isFinite(value);
  }
  if (typeof value !== "string") {
    return false;
  }
  var text = value.trim();
  if (text.length === 0) {
    return false;
  }
  return isFinite(Number(text));
}

function clearBuiltPayload(built) {
  if (!built) {
    return;
  }
  try {
    if (built.wrapper) {
      built.wrapper.clear();
    } else if (built.dict) {
      built.dict.clear();
    }
  } catch (err) {
    // Best-effort cleanup for transient Max Dict wrappers.
  }
}

function liveApiValueToJson(value, contextName) {
  if (value === undefined) {
    return safeJsonStringify(null, contextName + "_undefined");
  }
  // LiveAPI often returns arrays that include the property name; preserve them.
  if (Array.isArray(value)) {
    return safeJsonStringify(value, contextName + "_array");
  }
  if (value && typeof value === "object") {
    // Attempt to coerce Max Dict-like objects into plain data.
    try {
      if (value instanceof Dict) {
        var dictJson = value.stringify();
        return dictJson && dictJson.length > 0
          ? dictJson
          : safeJsonStringify({ dict: true }, contextName + "_dict_empty");
      }
    } catch (err) {
      // Fall through to generic object handling.
    }
    return safeJsonStringify(value, contextName + "_object");
  }
  return safeJsonStringify(value, contextName + "_scalar");
}

function escapeRegExp(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeCapabilityToken(token) {
  var value = token === undefined || token === null ? "" : String(token).trim();
  if (value.length === 0) {
    return "";
  }
  value = value.replace(/\(.*\)$/, "");
  value = value.replace(/[^A-Za-z0-9_]/g, "");
  return value;
}

function readApiInfoText(api) {
  if (!api) {
    return "";
  }
  try {
    if (api.info === undefined || api.info === null) {
      return "";
    }
    return String(api.info);
  } catch (err) {
    return "";
  }
}

function parseApiCapabilities(infoText) {
  var parsed = {
    info: String(infoText || ""),
    properties: {},
    functions: {},
    children: {},
    hasPropertiesList: false,
    hasFunctionsList: false,
    hasChildrenList: false,
  };
  if (!parsed.info) {
    return parsed;
  }

  var lines = parsed.info.split(/\r?\n|;/);
  for (var i = 0; i < lines.length; i += 1) {
    var line = String(lines[i] || "").trim();
    if (!line) {
      continue;
    }
    var m = line.match(/^(children|properties|functions)\s*:?\s*(.*)$/i);
    if (!m) {
      continue;
    }
    var bucket = String(m[1] || "").toLowerCase();
    var rest = String(m[2] || "");
    var tokens = rest.split(/[,\s]+/);
    if (bucket === "properties") {
      parsed.hasPropertiesList = true;
    } else if (bucket === "functions") {
      parsed.hasFunctionsList = true;
    } else if (bucket === "children") {
      parsed.hasChildrenList = true;
    }
    for (var t = 0; t < tokens.length; t += 1) {
      var token = normalizeCapabilityToken(tokens[t]);
      if (!token) {
        continue;
      }
      if (bucket === "properties") {
        parsed.properties[token] = true;
      } else if (bucket === "functions") {
        parsed.functions[token] = true;
      } else if (bucket === "children") {
        parsed.children[token] = true;
      }
    }
  }
  return parsed;
}

function getApiCapabilities(api) {
  return parseApiCapabilities(readApiInfoText(api));
}

function ensureInitialized(requestId) {
  var currentId = song ? Number(song.id) : 0;
  if (initialized && song && currentId > 0) {
    return true;
  }
  init(requestId);
  currentId = song ? Number(song.id) : 0;
  if (initialized && song && currentId > 0) {
    return true;
  }
  debug("LiveAPI not initialized yet or not attached to live_set.");
  ackWithRequest("error", ["not_initialized"], requestId);
  return false;
}

function init(requestId) {
  try {
    song = new LiveAPI(null, "live_set");
    var id = song ? Number(song.id) : 0;
    if (!(id > 0)) {
      initialized = false;
      debug("LiveAPI attached to live_set but id is invalid: " + id);
      ackWithRequest("error", ["not_in_live_set", id], requestId);
      return;
    }
    initialized = true;
    debug("Initialized LiveAPI at path: " + song.path + " (id=" + id + ")");
    ack("ack", "ready", song.path, id);
  } catch (err) {
    initialized = false;
    debug("Failed to initialize LiveAPI: " + err);
  }
}

function loadbang() {
  // LiveAPI must not be created in global scope; defer initialization.
  // The patch should also send an explicit init via live.thisdevice -> deferlow.
  init();
}

function ping() {
  ack("ack", "pong");
}

function api_ping(requestId) {
  if (!ensureInitialized(requestId)) return;
  ackWithRequest("pong", [], requestId);
}

function api_get(path, property, requestId) {
  if (!ensureInitialized(requestId)) return;
  var contextName = "api_get";
  var api = resolveApiOrError(path, contextName, requestId);
  if (!api) return;

  var propName = property === undefined || property === null ? "" : String(property).trim();
  if (propName.length === 0) {
    ackWithRequest("error", ["api_missing_property", String(path || "")], requestId);
    return;
  }
  var capabilities = getApiCapabilities(api);
  if (capabilities.hasPropertiesList && !capabilities.properties[propName]) {
    ackWithRequest("error", ["api_unknown_property", api.path, propName], requestId);
    return;
  }

  var rawValue = null;
  try {
    rawValue = api.get(propName);
  } catch (err) {
    debug("LiveAPI get failed for " + api.path + "." + propName + ": " + err);
    ackWithRequest("error", ["api_get_failed", api.path, propName], requestId);
    return;
  }

  var valueJson = liveApiValueToJson(rawValue, contextName + "_" + propName);
  ackWithRequest("api_get", [api.path, propName, valueJson], requestId);
}

function api_set(authToken, path, property, valueJson, requestId) {
  if (!requireMutationAuth("api_set", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var contextName = "api_set";
  var api = resolveApiOrError(path, contextName, requestId);
  if (!api) return;

  var propName = property === undefined || property === null ? "" : String(property).trim();
  if (propName.length === 0) {
    ackWithRequest("error", ["api_missing_property", String(path || "")], requestId);
    return;
  }
  var capabilities = getApiCapabilities(api);
  if (capabilities.hasPropertiesList && !capabilities.properties[propName]) {
    ackWithRequest("error", ["api_unknown_property", api.path, propName], requestId);
    return;
  }

  if (valueJson === undefined || valueJson === null || String(valueJson).length === 0) {
    ackWithRequest("error", ["api_missing_value", api.path, propName], requestId);
    return;
  }

  var parsedValue = parseJsonPayload(valueJson, contextName + "_" + propName, null, requestId);
  if (parsedValue === null && !(String(valueJson) === "null")) {
    // parseJsonPayload already acknowledged the error.
    return;
  }

  try {
    api.set(propName, parsedValue);
  } catch (err) {
    debug("LiveAPI set failed for " + api.path + "." + propName + ": " + err);
    ackWithRequest("error", ["api_set_failed", api.path, propName], requestId);
    return;
  }

  var resultJson = safeJsonStringify({ ok: true }, contextName + "_result");
  ackWithRequest("api_set", [api.path, propName, resultJson], requestId);
}

function api_call(authToken, path, method, argsJson, requestId) {
  if (!requireMutationAuth("api_call", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var contextName = "api_call";
  var api = resolveApiOrError(path, contextName, requestId);
  if (!api) return;
  var startedMs = new Date().getTime();

  var methodName = method === undefined || method === null ? "" : String(method).trim();
  if (methodName.length === 0) {
    ackWithRequest("error", ["api_missing_method", api.path], requestId);
    return;
  }
  var capabilities = getApiCapabilities(api);
  if (capabilities.hasFunctionsList && !capabilities.functions[methodName]) {
    ackWithRequest("error", ["api_unknown_method", api.path, methodName], requestId);
    return;
  }

  var parsedArgs = parseJsonPayload(argsJson, contextName + "_" + methodName, [], requestId);
  if (parsedArgs === null) {
    return;
  }
  var argsArray = normalizeArgsArray(parsedArgs);

  // LiveAPI's add_new_notes requires a Max Dict rather than a plain JS object.
  // Allow /api/call ... add_new_notes {"notes":[...]} by converting here.
  var builtPayload = null;
  if (methodName === "add_new_notes") {
    var notesPayload = argsArray.length > 0 ? argsArray[0] : null;
    var notesList = null;
    if (notesPayload && typeof notesPayload === "object") {
      if (Array.isArray(notesPayload)) {
        notesList = notesPayload;
      } else if (Array.isArray(notesPayload.notes)) {
        notesList = notesPayload.notes;
      }
    }
    if (!notesList || notesList.length === 0) {
      ackWithRequest("error", ["api_add_new_notes_invalid_payload", api.path], requestId);
      return;
    }
    builtPayload = buildNotesDict(notesList, contextName + "_add_new_notes", requestId);
    if (!builtPayload) {
      return;
    }
    argsArray[0] = builtPayload.dict;
  } else if (
    methodName === "apply_note_modifications" ||
    methodName === "remove_notes_extended" ||
    methodName === "get_notes_extended"
  ) {
    var payload = argsArray.length > 0 ? argsArray[0] : null;
    if (!payload || typeof payload !== "object") {
      ackWithRequest("error", ["api_" + methodName + "_invalid_payload", api.path], requestId);
      return;
    }
    builtPayload = buildGenericDict(payload, contextName + "_" + methodName, requestId);
    if (!builtPayload) {
      return;
    }
    argsArray[0] = builtPayload.dict;
  }

  var result = null;
  try {
    result = api.call.apply(api, [methodName].concat(argsArray));
  } catch (err) {
    debug("LiveAPI call failed for " + api.path + "." + methodName + ": " + err);
    ackWithRequest("error", ["api_call_failed", api.path, methodName], requestId);
    return;
  } finally {
    clearBuiltPayload(builtPayload);
  }

  var resultJson = liveApiValueToJson(result, contextName + "_" + methodName);
  debug(
    "api_call " + methodName + " elapsed_ms=" + (new Date().getTime() - startedMs)
  );
  ackWithRequest("api_call", [api.path, methodName, resultJson], requestId);
}

function api_children(path, childName, requestId) {
  if (!ensureInitialized(requestId)) return;
  var contextName = "api_children";
  var api = resolveApiOrError(path, contextName, requestId);
  if (!api) return;

  var childProp = childName === undefined || childName === null ? "" : String(childName).trim();
  if (childProp.length === 0) {
    ackWithRequest("error", ["api_missing_child_name", api.path], requestId);
    return;
  }

  var count = 0;
  try {
    count = api.getcount(childProp);
  } catch (err) {
    debug("LiveAPI getcount failed for " + api.path + "." + childProp + ": " + err);
    ackWithRequest("error", ["api_children_count_failed", api.path, childProp], requestId);
    return;
  }

  var children = [];
  var apiPath = normalizeLiveApiPath(api.path, path);
  for (var i = 0; i < count; i += 1) {
    var childPath = apiPath + " " + childProp + " " + i;
    try {
      var childApi = new LiveAPI(null, childPath);
      var childId = childApi ? Number(childApi.id) : 0;
      var childInfo = {
        index: i,
        id: childId,
        path: childPath,
      };
      try {
        var nameValue = getScalar(childApi, "name");
        if (nameValue !== undefined && nameValue !== null) {
          childInfo.name = String(nameValue);
        }
      } catch (errName) {}
      try {
        var typeValue = getScalar(childApi, "type");
        if (typeValue !== undefined && typeValue !== null) {
          childInfo.type = String(typeValue);
        }
      } catch (errType) {}
      children.push(childInfo);
    } catch (errChild) {
      debug("Failed to resolve child at " + childPath + ": " + errChild);
      children.push({ index: i, id: 0, path: childPath, error: "resolve_failed" });
    }
  }

  var childrenJson = safeJsonStringify(children, contextName + "_children");
  ackWithRequest("api_children", [apiPath, childProp, childrenJson], requestId);
}

function api_describe(path, requestId) {
  if (!ensureInitialized(requestId)) return;
  var contextName = "api_describe";
  var api = resolveApiOrError(path, contextName, requestId);
  if (!api) return;

  var describe = {
    path: api.path,
    id: api ? Number(api.id) : 0,
  };
  var capabilities = getApiCapabilities(api);
  try {
    var nameValue = getScalar(api, "name");
    if (nameValue !== undefined && nameValue !== null) {
      describe.name = String(nameValue);
    }
  } catch (errName) {}
  try {
    var typeValue = getScalar(api, "type");
    if (typeValue !== undefined && typeValue !== null) {
      describe.type = String(typeValue);
    }
  } catch (errType) {}
  if (capabilities.hasPropertiesList) {
    describe.properties = Object.keys(capabilities.properties);
  }
  if (capabilities.hasFunctionsList) {
    describe.functions = Object.keys(capabilities.functions);
  }
  if (capabilities.hasChildrenList) {
    describe.children = Object.keys(capabilities.children);
  }
  if (capabilities.info) {
    describe.info_excerpt = String(capabilities.info).slice(0, 600);
  }

  var describeJson = safeJsonStringify(describe, contextName + "_describe");
  ackWithRequest("api_describe", [api.path, describeJson], requestId);
}

function api_session_context(requestId) {
  if (!ensureInitialized(requestId)) return;
  var totalTracks = getTotalTracksOrError("session_context", requestId);
  if (totalTracks === 0) return;
  var midiTracks = countMidiTracks(totalTracks, "session_context", requestId);
  if (midiTracks === null) return;
  var audioTracks = countAudioTracks(totalTracks, "session_context", requestId);
  if (audioTracks === null) return;
  var payload = {
    generated_ms: nowMs(),
    song: readApiPropertyBag(song, [
      "tempo",
      "signature_numerator",
      "signature_denominator",
      "current_song_time",
      "is_playing",
      "clip_trigger_quantization",
      "metronome",
      "overdub",
      "record_mode",
      "session_record",
      "session_record_status",
      "re_enable_automation_enabled",
      "swing_amount",
      "song_length",
    ]),
    counts: {
      tracks: totalTracks,
      midi_tracks: midiTracks,
      audio_tracks: audioTracks,
      return_tracks: 0,
      scenes: 0,
    },
    selected: {
      track: describeApiTarget("live_set view selected_track", ["name"]),
      scene: describeApiTarget("live_set view selected_scene", ["name"]),
      device: describeApiTarget("live_set view selected_track view selected_device", ["name", "class_name"]),
    },
  };
  payload.song.path = String(song.path || "live_set");
  payload.song.id = Number(song.id || 0);
  try {
    payload.counts.return_tracks = song.getcount("return_tracks");
  } catch (errReturns) {}
  try {
    payload.counts.scenes = song.getcount("scenes");
  } catch (errScenes) {}
  ackWithRequest("api_session_context", [safeJsonStringify(payload, "session_context")], requestId);
}

function api_theory_status(requestId) {
  if (!ensureInitialized(requestId)) return;
  var payload = {
    path: String(song.path || "live_set"),
    id: Number(song.id || 0),
    theory: readApiPropertyBag(song, [
      "root_note",
      "scale_name",
      "scale_intervals",
      "scale_mode",
    ]),
  };
  ackWithRequest("api_theory_status", [safeJsonStringify(payload, "theory_status")], requestId);
}

function api_tuning_status(requestId) {
  if (!ensureInitialized(requestId)) return;
  var tuningApi = tryResolveApi("live_set tuning_system");
  var payload = {
    path: "live_set tuning_system",
    available: !!tuningApi,
  };
  if (tuningApi) {
    payload.id = Number(tuningApi.id || 0);
    payload.tuning = readApiPropertyBag(tuningApi, [
      "name",
      "pseudo_octave_in_cents",
      "lowest_note",
      "highest_note",
      "reference_pitch",
      "note_tunings",
    ]);
  }
  ackWithRequest("api_tuning_status", [safeJsonStringify(payload, "tuning_status")], requestId);
}

function api_device_list(trackRef, requestId) {
  if (!ensureInitialized(requestId)) return;
  var target = trackRef === undefined || trackRef === null ? "all" : String(trackRef).trim();
  var payload = { target: target, tracks: [] };
  if (target.length === 0 || target === "all") {
    var totalTracks = getTotalTracksOrError("device_list", requestId);
    if (totalTracks === 0) return;
    for (var i = 0; i < totalTracks; i += 1) {
      payload.tracks.push(describeDevicesForTrackPath("live_set tracks " + i));
    }
  } else {
    payload.tracks.push(describeDevicesForTrackPath(normalizeTrackPathReference(target, "live_set tracks 0")));
  }
  ackWithRequest("api_device_list", [target || "all", safeJsonStringify(payload, "device_list")], requestId);
}

function api_device_parameters(devicePath, requestId) {
  if (!ensureInitialized(requestId)) return;
  var pathText = devicePath === undefined || devicePath === null ? "" : String(devicePath).trim();
  if (pathText.length === 0) {
    ackWithRequest("error", ["api_missing_device_path"], requestId);
    return;
  }
  var payload = describeDeviceApi(tryResolveApi(pathText), pathText, true);
  ackWithRequest("api_device_parameters", [pathText, safeJsonStringify(payload, "device_parameters")], requestId);
}

function api_parameter_set(authToken, parameterPath, valueJson, requestId) {
  if (!requireMutationAuth("api_parameter_set", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var pathText = parameterPath === undefined || parameterPath === null ? "" : String(parameterPath).trim();
  if (pathText.length === 0) {
    ackWithRequest("error", ["api_missing_parameter_path"], requestId);
    return;
  }
  var parameterApi = resolveApiOrError(pathText, "parameter_set", requestId);
  if (!parameterApi) return;
  if (valueJson === undefined || valueJson === null || String(valueJson).trim().length === 0) {
    ackWithRequest("error", ["api_parameter_set_missing_value", pathText], requestId);
    return;
  }
  var parsedValue = parseJsonPayload(valueJson, "parameter_set_value", null, requestId);
  if (parsedValue === null && String(valueJson) !== "null") {
    return;
  }
  if (parsedValue === null) {
    ackWithRequest("error", ["api_parameter_set_missing_value", pathText], requestId);
    return;
  }
  if (!isNumericScalar(parsedValue)) {
    ackWithRequest("error", ["api_parameter_set_invalid_value_type", pathText, parsedValue], requestId);
    return;
  }
  var numericValue = Number(parsedValue);
  if (!isFinite(numericValue)) {
    ackWithRequest("error", ["api_parameter_set_invalid_value", pathText, parsedValue], requestId);
    return;
  }
  var before = describeParameterApi(parameterApi, pathText);
  if (Number(before.is_enabled) === 0) {
    ackWithRequest("error", ["api_parameter_disabled", pathText], requestId);
    return;
  }
  if (before.min !== undefined && before.min !== null && before.max !== undefined && before.max !== null) {
    var minValue = Number(before.min);
    var maxValue = Number(before.max);
    if (isFinite(minValue) && isFinite(maxValue) && (numericValue < minValue || numericValue > maxValue)) {
      ackWithRequest("error", ["api_parameter_value_out_of_range", pathText, numericValue, minValue, maxValue], requestId);
      return;
    }
  }
  try {
    parameterApi.set("value", numericValue);
  } catch (err) {
    debug("Failed to set parameter " + pathText + ": " + err);
    ackWithRequest("error", ["api_parameter_set_failed", pathText], requestId);
    return;
  }
  var payload = describeParameterApi(parameterApi, pathText);
  ackWithRequest("api_parameter_set", [pathText, safeJsonStringify(payload, "parameter_set")], requestId);
}

function api_mixer_status(trackRef, requestId) {
  if (!ensureInitialized(requestId)) return;
  var trackPath = normalizeTrackPathReference(trackRef, "live_set tracks 0");
  var mixerPath = trackPath + " mixer_device";
  var mixerApi = tryResolveApi(mixerPath);
  if (!mixerApi) {
    ackWithRequest("error", ["api_mixer_status_failed", trackPath], requestId);
    return;
  }
  var resolvedMixerPath = normalizeLiveApiPath(mixerApi.path, mixerPath);
  var payload = {
    track_path: trackPath,
    mixer_path: resolvedMixerPath,
    mixer: readApiPropertyBag(mixerApi, ["crossfade_assign", "panning_mode"]),
    parameters: {
      volume: describeParameterPath(resolvedMixerPath + " volume"),
      panning: describeParameterPath(resolvedMixerPath + " panning"),
      track_activator: describeParameterPath(resolvedMixerPath + " track_activator"),
    },
    sends: [],
  };
  var sendCount = 0;
  try {
    sendCount = mixerApi.getcount("sends");
  } catch (errSends) {
    sendCount = 0;
  }
  for (var i = 0; i < sendCount; i += 1) {
    payload.sends.push(describeParameterPath(resolvedMixerPath + " sends " + i));
  }
  ackWithRequest("api_mixer_status", [trackPath, safeJsonStringify(payload, "mixer_status")], requestId);
}

function api_insert_device(authToken, targetPath, deviceName, targetIndex, requestId) {
  if (!requireMutationAuth("api_insert_device", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var pathText = targetPath === undefined || targetPath === null ? "" : String(targetPath).trim();
  var nameText = deviceName === undefined || deviceName === null ? "" : String(deviceName).trim();
  if (pathText.length === 0 || nameText.length === 0) {
    ackWithRequest("error", ["api_insert_device_missing_args"], requestId);
    return;
  }
  var targetApi = resolveApiOrError(pathText, "insert_device", requestId);
  if (!targetApi) return;
  var resolvedTargetPath = normalizeLiveApiPath(targetApi.path, pathText);
  var capabilities = getApiCapabilities(targetApi);
  if (capabilities.hasFunctionsList && !capabilities.functions.insert_device) {
    ackWithRequest("error", ["api_insert_device_unsupported", resolvedTargetPath], requestId);
    return;
  }
  var parsedIndex = parseOptionalInsertionIndex(
    targetIndex,
    "api_insert_device_invalid_index",
    resolvedTargetPath,
    requestId
  );
  if (!parsedIndex.ok) return;
  var result = null;
  try {
    if (!parsedIndex.has_index) {
      result = targetApi.call("insert_device", nameText);
    } else {
      result = targetApi.call("insert_device", nameText, parsedIndex.value);
    }
  } catch (err) {
    debug("insert_device failed for " + resolvedTargetPath + " device=" + nameText + ": " + err);
    ackWithRequest("error", ["api_insert_device_failed", resolvedTargetPath, nameText], requestId);
    return;
  }
  var payload = {
    ok: true,
    target_path: resolvedTargetPath,
    device_name: nameText,
    target_index: parsedIndex.has_index ? parsedIndex.value : null,
    result: result,
  };
  ackWithRequest("api_insert_device", [resolvedTargetPath, nameText, safeJsonStringify(payload, "insert_device")], requestId);
}

function api_insert_chain(authToken, rackPath, targetIndex, requestId) {
  if (!requireMutationAuth("api_insert_chain", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var pathText = rackPath === undefined || rackPath === null ? "" : String(rackPath).trim();
  if (pathText.length === 0) {
    ackWithRequest("error", ["api_insert_chain_missing_path"], requestId);
    return;
  }
  var rackApi = resolveApiOrError(pathText, "insert_chain", requestId);
  if (!rackApi) return;
  var resolvedRackPath = normalizeLiveApiPath(rackApi.path, pathText);
  var capabilities = getApiCapabilities(rackApi);
  if (capabilities.hasFunctionsList && !capabilities.functions.insert_chain) {
    ackWithRequest("error", ["api_insert_chain_unsupported", resolvedRackPath], requestId);
    return;
  }
  var parsedIndex = parseOptionalInsertionIndex(
    targetIndex,
    "api_insert_chain_invalid_index",
    resolvedRackPath,
    requestId
  );
  if (!parsedIndex.ok) return;
  var result = null;
  try {
    if (parsedIndex.has_index) {
      result = rackApi.call("insert_chain", parsedIndex.value);
    } else {
      result = rackApi.call("insert_chain");
    }
  } catch (err) {
    debug("insert_chain failed for " + resolvedRackPath + ": " + err);
    ackWithRequest("error", ["api_insert_chain_failed", resolvedRackPath], requestId);
    return;
  }
  var payload = {
    ok: true,
    rack_path: resolvedRackPath,
    target_index: parsedIndex.has_index ? parsedIndex.value : null,
    result: result,
  };
  ackWithRequest("api_insert_chain", [resolvedRackPath, safeJsonStringify(payload, "insert_chain")], requestId);
}

function api_drum_chain_in_note(authToken, chainPath, noteValue, requestId) {
  if (!requireMutationAuth("api_drum_chain_in_note", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var pathText = chainPath === undefined || chainPath === null ? "" : String(chainPath).trim();
  var note = Number(noteValue);
  if (pathText.length === 0 || !isFinite(note) || Math.floor(note) !== note || note < -1 || note > 127) {
    ackWithRequest("error", ["api_drum_chain_in_note_invalid_args", pathText, noteValue], requestId);
    return;
  }
  var chainApi = resolveApiOrError(pathText, "drum_chain_in_note", requestId);
  if (!chainApi) return;
  var resolvedChainPath = normalizeLiveApiPath(chainApi.path, pathText);
  try {
    chainApi.set("in_note", note);
  } catch (err) {
    debug("drum_chain_in_note failed for " + resolvedChainPath + ": " + err);
    ackWithRequest("error", ["api_drum_chain_in_note_failed", resolvedChainPath, note], requestId);
    return;
  }
  var payload = readApiPropertyBag(chainApi, ["in_note", "out_note", "choke_group", "name"]);
  var hasAppliedNote = Object.prototype.hasOwnProperty.call(payload, "in_note");
  var appliedNote = hasAppliedNote ? Number(payload.in_note) : NaN;
  if (
    !hasAppliedNote ||
    payload.in_note === null ||
    payload.in_note === undefined ||
    String(payload.in_note).trim().length === 0 ||
    (payload.errors && payload.errors.in_note) ||
    !isFinite(appliedNote) ||
    Math.floor(appliedNote) !== appliedNote
  ) {
    ackWithRequest(
      "error",
      ["api_drum_chain_in_note_readback_failed", resolvedChainPath, note],
      requestId
    );
    return;
  }
  if (appliedNote !== note) {
    ackWithRequest(
      "error",
      ["api_drum_chain_in_note_write_not_applied", resolvedChainPath, note, appliedNote],
      requestId
    );
    return;
  }
  payload.path = resolvedChainPath;
  payload.id = Number(chainApi.id || 0);
  ackWithRequest("api_drum_chain_in_note", [resolvedChainPath, safeJsonStringify(payload, "drum_chain_in_note")], requestId);
}

function utf8ByteLength(value) {
  var text = String(value);
  var length = 0;
  for (var i = 0; i < text.length; i += 1) {
    var code = text.charCodeAt(i);
    if (code <= 0x7f) {
      length += 1;
    } else if (code <= 0x7ff) {
      length += 2;
    } else if (code >= 0xd800 && code <= 0xdbff) {
      var next = i + 1 < text.length ? text.charCodeAt(i + 1) : 0;
      if (next >= 0xdc00 && next <= 0xdfff) {
        length += 4;
        i += 1;
      } else {
        length += 3;
      }
    } else {
      length += 3;
    }
  }
  return length;
}

function truncateUtf8ToByteLength(value, maxBytes) {
  var text = String(value);
  var bounded = "";
  var length = 0;
  for (var i = 0; i < text.length; i += 1) {
    var code = text.charCodeAt(i);
    var chunk = text.charAt(i);
    var chunkLength = 0;
    if (code <= 0x7f) {
      chunkLength = 1;
    } else if (code <= 0x7ff) {
      chunkLength = 2;
    } else if (code >= 0xd800 && code <= 0xdbff) {
      var next = i + 1 < text.length ? text.charCodeAt(i + 1) : 0;
      if (next >= 0xdc00 && next <= 0xdfff) {
        chunk += text.charAt(i + 1);
        chunkLength = 4;
        i += 1;
      } else {
        chunkLength = 3;
      }
    } else {
      chunkLength = 3;
    }
    if (length + chunkLength > maxBytes) {
      break;
    }
    bounded += chunk;
    length += chunkLength;
  }
  return bounded;
}

function oscStringEncodedByteLength(value) {
  var lengthWithNull = utf8ByteLength(value) + 1;
  var remainder = lengthWithNull % 4;
  return remainder === 0 ? lengthWithNull : lengthWithNull + (4 - remainder);
}

function sessionClipInspectionAckPacketByteLength(fragmentJson, requestId) {
  return (
    oscStringEncodedByteLength("/ack") +
    oscStringEncodedByteLength(",sss") +
    oscStringEncodedByteLength("api_session_clip_inspect") +
    oscStringEncodedByteLength(fragmentJson) +
    oscStringEncodedByteLength(requestId)
  );
}

function sessionClipInspectionError(category, details, requestId) {
  var boundedDetails = (details || []).map(function (detail) {
    if (typeof detail === "number" && isFinite(detail)) {
      return detail;
    }
    return truncateUtf8ToByteLength(
      detail === undefined ? "undefined" : detail === null ? "null" : String(detail),
      256
    );
  });
  ackWithRequest(
    "error",
    ["api_session_clip_inspect_" + category].concat(boundedDetails),
    truncateUtf8ToByteLength(
      requestId === undefined || requestId === null ? "" : String(requestId),
      128
    )
  );
}

function newSessionClipInspectionId() {
  sessionClipInspectionCounter += 1;
  return "session_clip_" + nowMs() + "_" + sessionClipInspectionCounter;
}

function isNonNegativeInteger(value) {
  return (
    typeof value === "number" &&
    isFinite(value) &&
    value >= 0 &&
    Math.floor(value) === value
  );
}

function readSessionClipInspectionProperty(api, property, target, requestId) {
  try {
    var value = getScalar(api, property);
    if (value === undefined || value === null) {
      sessionClipInspectionError("read_failed", [target, property], requestId);
      return { ok: false, value: null };
    }
    return { ok: true, value: value };
  } catch (err) {
    debug(
      "Session clip inspection failed to read " +
        target +
        "." +
        property +
        ": " +
        err
    );
    sessionClipInspectionError("read_failed", [target, property], requestId);
    return { ok: false, value: null };
  }
}

function readNullableSessionClipInspectionTextProperty(api, property) {
  try {
    var value = getScalar(api, property);
    return typeof value === "string" ? value : null;
  } catch (err) {
    return null;
  }
}

function readNullableSessionClipInspectionDeviceType(api) {
  try {
    var value = getScalar(api, "type");
    if (
      typeof value === "number" &&
      isFinite(value) &&
      Math.floor(value) === value &&
      (value === 0 || value === 1 || value === 2 || value === 4)
    ) {
      return value;
    }
  } catch (err) {
    // Device type is optional metadata.
  }
  return null;
}

function validateSessionClipInspectionClipData(clipData, requestId) {
  var positionFields = [
    "start_marker",
    "end_marker",
    "loop_start",
    "loop_end",
  ];
  for (var i = 0; i < positionFields.length; i += 1) {
    var positionField = positionFields[i];
    if (
      typeof clipData[positionField] !== "number" ||
      !isFinite(clipData[positionField])
    ) {
      sessionClipInspectionError(
        "parse_failed",
        ["clip_invalid_field", positionField],
        requestId
      );
      return false;
    }
  }
  if (
    typeof clipData.live_length !== "number" ||
    !isFinite(clipData.live_length) ||
    clipData.live_length < 0
  ) {
    sessionClipInspectionError(
      "parse_failed",
      ["clip_invalid_field", "live_length"],
      requestId
    );
    return false;
  }
  if (
    clipData.start_marker > clipData.end_marker ||
    !isFinite(clipData.end_marker - clipData.start_marker) ||
    clipData.loop_start > clipData.loop_end ||
    !isFinite(clipData.loop_end - clipData.loop_start)
  ) {
    sessionClipInspectionError(
      "parse_failed",
      ["clip_invalid_ranges"],
      requestId
    );
    return false;
  }
  return true;
}

function openSessionClipInspectionApi(path, target, requestId) {
  try {
    var api = new LiveAPI(null, path);
    if (!api || !(Number(api.id) > 0)) {
      sessionClipInspectionError("not_found", [target, path], requestId);
      return null;
    }
    return api;
  } catch (err) {
    debug("Session clip inspection could not resolve " + target + " at " + path + ": " + err);
    sessionClipInspectionError("not_found", [target, path], requestId);
    return null;
  }
}

function parseSessionClipInspectionNotes(rawResult, requestId) {
  var parsed = null;
  if (rawResult && typeof rawResult === "object" && !Array.isArray(rawResult)) {
    parsed = rawResult;
  } else {
    var rawText = Array.isArray(rawResult)
      ? rawResult.join(" ")
      : rawResult === undefined || rawResult === null
        ? ""
        : String(rawResult);
    try {
      parsed = JSON.parse(rawText);
    } catch (err) {
      debug("Session clip inspection note JSON parse failed: " + err);
      sessionClipInspectionError("parse_failed", ["notes"], requestId);
      return null;
    }
  }
  if (!parsed || !Array.isArray(parsed.notes)) {
    sessionClipInspectionError("parse_failed", ["notes_shape"], requestId);
    return null;
  }
  return parsed.notes;
}

function copySessionClipInspectionNote(note, index, requestId) {
  if (!note || typeof note !== "object" || Array.isArray(note)) {
    sessionClipInspectionError("parse_failed", ["note", index], requestId);
    return null;
  }
  var fields = [
    "note_id",
    "pitch",
    "start_time",
    "duration",
    "velocity",
    "mute",
    "probability",
    "velocity_deviation",
    "release_velocity",
  ];
  var copied = {};
  for (var i = 0; i < fields.length; i += 1) {
    var field = fields[i];
    if (!Object.prototype.hasOwnProperty.call(note, field) || note[field] === undefined) {
      sessionClipInspectionError(
        "parse_failed",
        ["note_missing_field", index, field],
        requestId
      );
      return null;
    }
    copied[field] = note[field];
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && isFinite(value);
  }

  if (
    !isFiniteNumber(copied.note_id) ||
    copied.note_id < 0 ||
    Math.floor(copied.note_id) !== copied.note_id
  ) {
    sessionClipInspectionError("parse_failed", ["note_invalid_field", index, "note_id"], requestId);
    return null;
  }
  if (
    !isFiniteNumber(copied.pitch) ||
    copied.pitch < 0 ||
    copied.pitch > 127 ||
    Math.floor(copied.pitch) !== copied.pitch
  ) {
    sessionClipInspectionError("parse_failed", ["note_invalid_field", index, "pitch"], requestId);
    return null;
  }
  if (!isFiniteNumber(copied.start_time)) {
    sessionClipInspectionError("parse_failed", ["note_invalid_field", index, "start_time"], requestId);
    return null;
  }
  if (
    !isFiniteNumber(copied.duration) ||
    copied.duration < 0 ||
    !isFinite(copied.start_time + copied.duration)
  ) {
    sessionClipInspectionError("parse_failed", ["note_invalid_field", index, "duration"], requestId);
    return null;
  }
  if (
    !isFiniteNumber(copied.velocity) ||
    copied.velocity < 0 ||
    copied.velocity > 127
  ) {
    sessionClipInspectionError("parse_failed", ["note_invalid_field", index, "velocity"], requestId);
    return null;
  }
  if (
    typeof copied.mute !== "boolean" &&
    !(
      isFiniteNumber(copied.mute) &&
      (copied.mute === 0 || copied.mute === 1)
    )
  ) {
    sessionClipInspectionError("parse_failed", ["note_invalid_field", index, "mute"], requestId);
    return null;
  }
  if (
    !isFiniteNumber(copied.probability) ||
    copied.probability < 0 ||
    copied.probability > 1
  ) {
    sessionClipInspectionError("parse_failed", ["note_invalid_field", index, "probability"], requestId);
    return null;
  }
  if (
    !isFiniteNumber(copied.velocity_deviation) ||
    copied.velocity_deviation < -127 ||
    copied.velocity_deviation > 127
  ) {
    sessionClipInspectionError(
      "parse_failed",
      ["note_invalid_field", index, "velocity_deviation"],
      requestId
    );
    return null;
  }
  if (
    !isFiniteNumber(copied.release_velocity) ||
    copied.release_velocity < 0 ||
    copied.release_velocity > 127
  ) {
    sessionClipInspectionError(
      "parse_failed",
      ["note_invalid_field", index, "release_velocity"],
      requestId
    );
    return null;
  }
  return copied;
}

function buildSessionClipInspectionSummary(notes) {
  var pitchMin = null;
  var pitchMax = null;
  for (var i = 0; i < notes.length; i += 1) {
    if (!Object.prototype.hasOwnProperty.call(notes[i], "pitch")) {
      continue;
    }
    var pitch = Number(notes[i].pitch);
    if (!isFinite(pitch)) {
      continue;
    }
    if (pitchMin === null || pitch < pitchMin) pitchMin = pitch;
    if (pitchMax === null || pitch > pitchMax) pitchMax = pitch;
  }
  return {
    note_count: notes.length,
    pitch_min: pitchMin,
    pitch_max: pitchMax,
  };
}

function makeSessionClipInspectionFragment(
  metadata,
  fragmentIndex,
  fragmentCount,
  fragmentKind,
  isLast,
  data
) {
  return {
    schema: SESSION_CLIP_INSPECTION_SCHEMA,
    schema_version: SESSION_CLIP_INSPECTION_SCHEMA_VERSION,
    producer_version: SESSION_CLIP_INSPECTION_PRODUCER_VERSION,
    inspection_id: metadata.inspection_id,
    correlation: metadata.correlation,
    snapshot: metadata.snapshot,
    transfer: {
      fragment_index: fragmentIndex,
      fragment_count: fragmentCount,
      fragment_kind: fragmentKind,
      is_last: !!isLast,
      packet_budget_bytes: SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES,
    },
    completeness: {
      track: "complete",
      clip: "complete",
      devices: "complete",
      notes: "complete",
      missing_fields: [],
    },
    data: data,
  };
}

function serializeSessionClipInspectionFragment(fragment) {
  try {
    var json = JSON.stringify(fragment);
    if (typeof json !== "string") {
      return { ok: false, json: "", packet_bytes: 0 };
    }
    return { ok: true, json: json, packet_bytes: 0 };
  } catch (err) {
    debug("Session clip inspection fragment serialization failed: " + err);
    return { ok: false, json: "", packet_bytes: 0 };
  }
}

function measureSessionClipInspectionFragment(
  metadata,
  fragmentIndex,
  fragmentCount,
  fragmentKind,
  isLast,
  data,
  requestId
) {
  var fragment = makeSessionClipInspectionFragment(
    metadata,
    fragmentIndex,
    fragmentCount,
    fragmentKind,
    isLast,
    data
  );
  var serialized = serializeSessionClipInspectionFragment(fragment);
  if (!serialized.ok) {
    return serialized;
  }
  serialized.packet_bytes = sessionClipInspectionAckPacketByteLength(
    serialized.json,
    requestId
  );
  return serialized;
}

function paginateSessionClipInspectionItems(
  metadata,
  items,
  fragmentKind,
  offsetField,
  countField,
  totalField,
  itemsField,
  requestId,
  maxFragmentCount,
  maxPages
) {
  var pages = [];
  var offset = 0;
  while (offset < items.length) {
    if (pages.length >= maxPages) {
      return {
        ok: false,
        error: "limit_exceeded",
        details: ["fragments", maxFragmentCount],
        pages: [],
      };
    }
    var pageItems = [];
    while (offset + pageItems.length < items.length) {
      var candidateItems = pageItems.concat([items[offset + pageItems.length]]);
      var candidateData = {};
      candidateData[offsetField] = offset;
      candidateData[countField] = candidateItems.length;
      candidateData[totalField] = items.length;
      candidateData[itemsField] = candidateItems;
      var measured = measureSessionClipInspectionFragment(
        metadata,
        Math.max(0, maxFragmentCount - 1),
        maxFragmentCount,
        fragmentKind,
        false,
        candidateData,
        requestId
      );
      if (!measured.ok) {
        return {
          ok: false,
          error: "serialization_failed",
          details: [fragmentKind, offset + pageItems.length],
          pages: [],
        };
      }
      if (measured.packet_bytes > SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES) {
        if (pageItems.length === 0) {
          return {
            ok: false,
            error: "item_too_large",
            details: [fragmentKind, offset, measured.packet_bytes],
            pages: [],
          };
        }
        break;
      }
      pageItems = candidateItems;
    }
    var pageData = {};
    pageData[offsetField] = offset;
    pageData[countField] = pageItems.length;
    pageData[totalField] = items.length;
    pageData[itemsField] = pageItems;
    pages.push(pageData);
    offset += pageItems.length;
  }
  return { ok: true, pages: pages };
}

function buildSessionClipInspectionFragments(metadata, contextData, devices, notes, requestId) {
  if (devices.length > SESSION_CLIP_INSPECTION_MAX_DEVICES) {
    return {
      ok: false,
      error: "limit_exceeded",
      details: ["devices", devices.length, SESSION_CLIP_INSPECTION_MAX_DEVICES],
      fragments: [],
    };
  }
  if (notes.length > SESSION_CLIP_INSPECTION_MAX_NOTES) {
    return {
      ok: false,
      error: "limit_exceeded",
      details: ["notes", notes.length, SESSION_CLIP_INSPECTION_MAX_NOTES],
      fragments: [],
    };
  }
  var completeData = {
    context: contextData.context,
    track: contextData.track,
    clip: contextData.clip,
    summary: contextData.summary,
    device_offset: 0,
    device_count: devices.length,
    device_total: devices.length,
    devices: devices,
    note_offset: 0,
    note_count: notes.length,
    note_total: notes.length,
    notes: notes,
  };
  var complete = measureSessionClipInspectionFragment(
    metadata,
    0,
    1,
    "complete",
    true,
    completeData,
    requestId
  );
  if (!complete.ok) {
    return { ok: false, error: "serialization_failed", details: ["complete"], fragments: [] };
  }
  if (complete.packet_bytes <= SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES) {
    return { ok: true, fragments: [complete.json] };
  }

  var maxFragmentCount = SESSION_CLIP_INSPECTION_MAX_FRAGMENTS;
  var contextMeasured = measureSessionClipInspectionFragment(
    metadata,
    Math.max(0, maxFragmentCount - 1),
    maxFragmentCount,
    "context",
    false,
    contextData,
    requestId
  );
  if (!contextMeasured.ok) {
    return { ok: false, error: "serialization_failed", details: ["context"], fragments: [] };
  }
  if (contextMeasured.packet_bytes > SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES) {
    return {
      ok: false,
      error: "item_too_large",
      details: ["context", 0, contextMeasured.packet_bytes],
      fragments: [],
    };
  }

  var devicePages = paginateSessionClipInspectionItems(
    metadata,
    devices,
    "device_page",
    "device_offset",
    "device_count",
    "device_total",
    "devices",
    requestId,
    maxFragmentCount,
    maxFragmentCount - 1
  );
  if (!devicePages.ok) {
    return {
      ok: false,
      error: devicePages.error,
      details: devicePages.details,
      fragments: [],
    };
  }
  var notePages = paginateSessionClipInspectionItems(
    metadata,
    notes,
    "note_page",
    "note_offset",
    "note_count",
    "note_total",
    "notes",
    requestId,
    maxFragmentCount,
    maxFragmentCount - 1 - devicePages.pages.length
  );
  if (!notePages.ok) {
    return {
      ok: false,
      error: notePages.error,
      details: notePages.details,
      fragments: [],
    };
  }

  var pageSpecs = [{ kind: "context", data: contextData }]
    .concat(
      devicePages.pages.map(function (page) {
        return { kind: "device_page", data: page };
      })
    )
    .concat(
      notePages.pages.map(function (page) {
        return { kind: "note_page", data: page };
      })
    );
  var fragmentCount = pageSpecs.length;
  if (fragmentCount > SESSION_CLIP_INSPECTION_MAX_FRAGMENTS) {
    return {
      ok: false,
      error: "limit_exceeded",
      details: ["fragments", fragmentCount, SESSION_CLIP_INSPECTION_MAX_FRAGMENTS],
      fragments: [],
    };
  }
  var fragments = [];
  for (var i = 0; i < pageSpecs.length; i += 1) {
    var measured = measureSessionClipInspectionFragment(
      metadata,
      i,
      fragmentCount,
      pageSpecs[i].kind,
      i === fragmentCount - 1,
      pageSpecs[i].data,
      requestId
    );
    if (!measured.ok) {
      return {
        ok: false,
        error: "serialization_failed",
        details: [pageSpecs[i].kind, i],
        fragments: [],
      };
    }
    if (measured.packet_bytes > SESSION_CLIP_INSPECTION_PACKET_BUDGET_BYTES) {
      return {
        ok: false,
        error: "item_too_large",
        details: [pageSpecs[i].kind, i, measured.packet_bytes],
        fragments: [],
      };
    }
    fragments.push(measured.json);
  }
  return { ok: true, fragments: fragments };
}

function api_session_clip_inspect(trackIndex, slotIndex, schemaVersion, requestId) {
  var requestText =
    requestId === undefined || requestId === null ? "" : String(requestId);
  if (requestText.length === 0) {
    sessionClipInspectionError("validation_failed", ["request_id_required"], requestText);
    return;
  }
  var requestByteLength = utf8ByteLength(requestText);
  if (requestByteLength > 128) {
    sessionClipInspectionError(
      "validation_failed",
      ["request_id_too_long", requestByteLength],
      truncateUtf8ToByteLength(requestText, 128)
    );
    return;
  }
  if (!isNonNegativeInteger(trackIndex)) {
    sessionClipInspectionError(
      "validation_failed",
      ["invalid_track_index", trackIndex],
      requestText
    );
    return;
  }
  if (!isNonNegativeInteger(slotIndex)) {
    sessionClipInspectionError(
      "validation_failed",
      ["invalid_slot_index", slotIndex],
      requestText
    );
    return;
  }
  if (
    typeof schemaVersion !== "number" ||
    schemaVersion !== SESSION_CLIP_INSPECTION_SCHEMA_VERSION
  ) {
    sessionClipInspectionError(
      "validation_failed",
      ["unsupported_schema_version", schemaVersion],
      requestText
    );
    return;
  }
  if (!ensureInitialized(requestText)) return;

  var track = Number(trackIndex);
  var slot = Number(slotIndex);
  var startedMs = nowMs();
  var trackPath = "live_set tracks " + track;
  var trackApi = openSessionClipInspectionApi(trackPath, "track", requestText);
  if (!trackApi) return;
  trackPath = normalizeLiveApiPath(trackApi.path, trackPath);

  var midiResult = readSessionClipInspectionProperty(
    trackApi,
    "has_midi_input",
    "track",
    requestText
  );
  if (!midiResult.ok) return;
  if (Number(midiResult.value) !== 1) {
    sessionClipInspectionError("not_midi", [track, trackPath], requestText);
    return;
  }
  var trackName = readNullableSessionClipInspectionTextProperty(
    trackApi,
    "name"
  );

  var slotCount = 0;
  try {
    slotCount = Number(trackApi.getcount("clip_slots"));
  } catch (errSlotCount) {
    debug("Session clip inspection could not read clip slot count: " + errSlotCount);
    sessionClipInspectionError("read_failed", ["track", "clip_slots"], requestText);
    return;
  }
  if (!(isFinite(slotCount) && slotCount >= 0 && Math.floor(slotCount) === slotCount)) {
    sessionClipInspectionError("read_failed", ["track", "clip_slots"], requestText);
    return;
  }
  if (!(slot < slotCount)) {
    sessionClipInspectionError("not_found", ["clip_slot", track, slot], requestText);
    return;
  }

  var slotPath = trackPath + " clip_slots " + slot;
  var slotApi = openSessionClipInspectionApi(slotPath, "clip_slot", requestText);
  if (!slotApi) return;
  var hasClip = readSessionClipInspectionProperty(
    slotApi,
    "has_clip",
    "clip_slot",
    requestText
  );
  if (!hasClip.ok) return;
  if (Number(hasClip.value) !== 1) {
    sessionClipInspectionError("no_clip", [track, slot], requestText);
    return;
  }

  var clipPath = slotPath + " clip";
  var clipApi = openSessionClipInspectionApi(clipPath, "clip", requestText);
  if (!clipApi) return;
  clipPath = normalizeLiveApiPath(clipApi.path, clipPath);
  var initialClipId = Number(clipApi.id);

  var clipPropertyMap = [
    ["start_marker", "start_marker"],
    ["end_marker", "end_marker"],
    ["length", "live_length"],
    ["looping", "looping"],
    ["loop_start", "loop_start"],
    ["loop_end", "loop_end"],
  ];
  var clipData = {
    slot_index: slot,
    path: clipPath,
    id: initialClipId,
    name: readNullableSessionClipInspectionTextProperty(clipApi, "name"),
  };
  for (var clipPropIndex = 0; clipPropIndex < clipPropertyMap.length; clipPropIndex += 1) {
    var clipProperty = clipPropertyMap[clipPropIndex][0];
    var outputProperty = clipPropertyMap[clipPropIndex][1];
    var clipValue = readSessionClipInspectionProperty(
      clipApi,
      clipProperty,
      "clip",
      requestText
    );
    if (!clipValue.ok) return;
    clipData[outputProperty] =
      outputProperty === "looping" ? !!Number(clipValue.value) : clipValue.value;
  }
  if (!validateSessionClipInspectionClipData(clipData, requestText)) return;

  var deviceCount = 0;
  try {
    deviceCount = Number(trackApi.getcount("devices"));
  } catch (errDeviceCount) {
    debug("Session clip inspection could not read device count: " + errDeviceCount);
    sessionClipInspectionError("read_failed", ["track", "devices"], requestText);
    return;
  }
  if (!(isFinite(deviceCount) && deviceCount >= 0 && Math.floor(deviceCount) === deviceCount)) {
    sessionClipInspectionError("read_failed", ["track", "devices"], requestText);
    return;
  }
  if (deviceCount > SESSION_CLIP_INSPECTION_MAX_DEVICES) {
    sessionClipInspectionError(
      "limit_exceeded",
      ["devices", deviceCount, SESSION_CLIP_INSPECTION_MAX_DEVICES],
      requestText
    );
    return;
  }
  var devices = [];
  for (var deviceIndex = 0; deviceIndex < deviceCount; deviceIndex += 1) {
    var devicePath = trackPath + " devices " + deviceIndex;
    var deviceApi = null;
    try {
      deviceApi = new LiveAPI(null, devicePath);
    } catch (errDevice) {
      debug("Session clip inspection could not read device " + deviceIndex + ": " + errDevice);
      sessionClipInspectionError("read_failed", ["device", deviceIndex], requestText);
      return;
    }
    if (!deviceApi || !(Number(deviceApi.id) > 0)) {
      sessionClipInspectionError("read_failed", ["device", deviceIndex], requestText);
      return;
    }
    var device = {
      index: deviceIndex,
      path: normalizeLiveApiPath(deviceApi.path, devicePath),
      id: Number(deviceApi.id),
      name: readNullableSessionClipInspectionTextProperty(deviceApi, "name"),
      class_name: readNullableSessionClipInspectionTextProperty(
        deviceApi,
        "class_name"
      ),
      type: readNullableSessionClipInspectionDeviceType(deviceApi),
    };
    devices.push(device);
  }

  var rawNotes = null;
  try {
    rawNotes = clipApi.call("get_all_notes_extended");
  } catch (errNotes) {
    debug("Session clip inspection note read failed: " + errNotes);
    sessionClipInspectionError("read_failed", ["notes"], requestText);
    return;
  }
  var parsedNotes = parseSessionClipInspectionNotes(rawNotes, requestText);
  if (parsedNotes === null) return;
  if (parsedNotes.length > SESSION_CLIP_INSPECTION_MAX_NOTES) {
    sessionClipInspectionError(
      "limit_exceeded",
      ["notes", parsedNotes.length, SESSION_CLIP_INSPECTION_MAX_NOTES],
      requestText
    );
    return;
  }
  var notes = [];
  for (var noteIndex = 0; noteIndex < parsedNotes.length; noteIndex += 1) {
    var copiedNote = copySessionClipInspectionNote(
      parsedNotes[noteIndex],
      noteIndex,
      requestText
    );
    if (copiedNote === null) return;
    notes.push(copiedNote);
  }

  var metadata = {
    inspection_id: newSessionClipInspectionId(),
    correlation: {
      request_id: requestText,
      track_index: track,
      slot_index: slot,
    },
    snapshot: {
      started_ms: startedMs,
      completed_ms: nowMs(),
      atomic: false,
      consistent: true,
    },
  };
  var contextData = {
    context: "session",
    track: {
      index: track,
      path: trackPath,
      id: Number(trackApi.id),
      name: trackName,
    },
    clip: clipData,
    summary: buildSessionClipInspectionSummary(notes),
  };
  var built = buildSessionClipInspectionFragments(
    metadata,
    contextData,
    devices,
    notes,
    requestText
  );
  if (!built.ok) {
    sessionClipInspectionError(built.error, built.details, requestText);
    return;
  }

  var finalClipApi = null;
  try {
    finalClipApi = new LiveAPI(null, clipPath);
  } catch (errReread) {
    debug("Session clip inspection clip id reread failed: " + errReread);
    sessionClipInspectionError("read_failed", ["clip", "id_reread"], requestText);
    return;
  }
  var finalClipId = finalClipApi ? Number(finalClipApi.id) : 0;
  if (finalClipId !== initialClipId) {
    sessionClipInspectionError(
      "snapshot_changed",
      [initialClipId, finalClipId],
      requestText
    );
    return;
  }

  for (var fragmentIndex = 0; fragmentIndex < built.fragments.length; fragmentIndex += 1) {
    ackWithRequest(
      "api_session_clip_inspect",
      [built.fragments[fragmentIndex]],
      requestText
    );
  }
}

function api_observe(authToken, path, property, optionsJson, requestId) {
  if (!requireMutationAuth("api_observe", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var contextName = "api_observe";
  var api = resolveApiOrError(path, contextName, requestId);
  if (!api) return;

  var propName = property === undefined || property === null ? "" : String(property).trim();
  if (propName.length === 0) {
    ackWithRequest("error", ["api_missing_property", String(path || "")], requestId);
    return;
  }

  var capabilities = getApiCapabilities(api);
  var propKnown = !capabilities.hasPropertiesList || !!capabilities.properties[propName];
  var childKnown = !capabilities.hasChildrenList || !!capabilities.children[propName];
  if (capabilities.hasPropertiesList || capabilities.hasChildrenList) {
    if (!propKnown && !childKnown) {
      ackWithRequest("error", ["api_unknown_property", api.path, propName], requestId);
      return;
    }
  }

  var parsedOptions = parseJsonPayload(optionsJson, contextName + "_options", {}, requestId);
  if (parsedOptions === null) {
    return;
  }
  if (!parsedOptions || typeof parsedOptions !== "object" || Array.isArray(parsedOptions)) {
    parsedOptions = {};
  }

  var observerId = parsedOptions.observer_id === undefined || parsedOptions.observer_id === null
    ? newObserverId()
    : String(parsedOptions.observer_id).trim();
  if (observerId.length === 0) {
    observerId = newObserverId();
  }
  if (!isValidObserverId(observerId)) {
    ackWithRequest("error", ["api_invalid_observer_id", observerId], requestId);
    return;
  }
  var mode = normalizeObserverMode(parsedOptions.mode);
  var emitInitial = parsedOptions.emit_initial === false ? false : true;
  var minIntervalMs = Math.max(
    0,
    Math.floor(Number(parsedOptions.min_interval_ms || parsedOptions.throttle_ms || 0))
  );

  if (!hasObserverEntry(observerId) && Object.keys(apiObservers).length >= MAX_API_OBSERVERS) {
    ackWithRequest("error", ["api_observer_limit_reached", MAX_API_OBSERVERS], requestId);
    return;
  }
  clearObserverEntry(observerId);

  var observerApi = null;
  var apiPath = normalizeLiveApiPath(api.path, path);
  try {
    observerApi = new LiveAPI(buildObserverCallback(observerId), apiPath);
    observerApi.mode = mode;
    observerApi.property = propName;
  } catch (err) {
    debug("Failed to install observer for " + api.path + "." + propName + ": " + err);
    ackWithRequest("error", ["api_observe_failed", api.path, propName], requestId);
    return;
  }

  var entry = {
    observer_id: observerId,
    requested_path: apiPath,
    property: propName,
    mode: mode,
    min_interval_ms: minIntervalMs,
    dropped_events: 0,
    last_emit_ms: 0,
    created_ms: nowMs(),
    event_count: 0,
    api: observerApi,
  };
  apiObservers[observerId] = entry;

  var initialArgs = [];
  if (emitInitial) {
    try {
      initialArgs = [observerApi.get(propName)];
    } catch (errInitial) {
      initialArgs = [];
    }
  }
  var payload = buildObserverPayload(entry, initialArgs);
  var payloadJson = safeJsonStringify(payload, contextName + "_payload");
  ackWithRequest("api_observe", [observerId, normalizeLiveApiPath(observerApi.path, apiPath), propName, payloadJson], requestId);
}

function api_unobserve(authToken, observerId, requestId) {
  if (!requireMutationAuth("api_unobserve", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var key = observerId === undefined || observerId === null ? "" : String(observerId).trim();
  if (key.length === 0) {
    ackWithRequest("error", ["api_missing_observer_id"], requestId);
    return;
  }
  var removed = clearObserverEntry(key);
  if (!removed) {
    ackWithRequest("error", ["api_observer_not_found", key], requestId);
    return;
  }
  var resultJson = safeJsonStringify(
    {
      observer_id: key,
      requested_path: String(removed.requested_path || ""),
      property: String(removed.property || ""),
      removed: true,
    },
    "api_unobserve_result"
  );
  ackWithRequest(
    "api_unobserve",
    [key, resultJson],
    requestId
  );
}

function api_clear_observers(authToken, requestId) {
  if (!requireMutationAuth("api_clear_observers", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var cleared = clearAllObserverEntries();
  var resultJson = safeJsonStringify({ cleared: Number(cleared || 0) }, "api_clear_observers");
  ackWithRequest("api_clear_observers", [resultJson], requestId);
}

function api_observers(requestId) {
  if (!ensureInitialized(requestId)) return;
  var payloadJson = safeJsonStringify(listObserverEntries(), "api_observers");
  ackWithRequest("api_observers", [payloadJson], requestId);
}

var API_FALLBACK_HANDLERS = {
  "api_session_context": api_session_context,
  "api_theory_status": api_theory_status,
  "api_tuning_status": api_tuning_status,
  "api_device_list": api_device_list,
  "api_device_parameters": api_device_parameters,
  "api_parameter_set": api_parameter_set,
  "api_mixer_status": api_mixer_status,
  "api_insert_device": api_insert_device,
  "api_insert_chain": api_insert_chain,
  "api_drum_chain_in_note": api_drum_chain_in_note,
  "api_session_clip_inspect": api_session_clip_inspect,
};

var API_FALLBACK_REQUEST_ID_INDEXES = {
  "api_session_context": 0,
  "api_theory_status": 0,
  "api_tuning_status": 0,
  "api_device_list": 1,
  "api_device_parameters": 1,
  "api_parameter_set": 3,
  "api_mixer_status": 1,
  "api_insert_device": 4,
  "api_insert_chain": 3,
  "api_drum_chain_in_note": 3,
  "api_session_clip_inspect": 3,
};

function fallbackRequestId(targetName, args) {
  var requestIndex = API_FALLBACK_REQUEST_ID_INDEXES[targetName];
  if (
    requestIndex === undefined ||
    requestIndex < 0 ||
    requestIndex >= args.length
  ) {
    return "";
  }
  return args[requestIndex];
}

function anything() {
  var rawName = typeof messagename === "undefined" ? "" : String(messagename || "");
  if (!rawName || rawName.charAt(0) !== "/") {
    debug("Unhandled message: " + rawName);
    return;
  }
  var targetName = rawName.slice(1).replace(/\//g, "_");
  var target = API_FALLBACK_HANDLERS[targetName];
  if (typeof target !== "function") {
    debug("Unhandled OSC selector: " + rawName);
    ack("ack", "error", "unknown_selector", rawName);
    return;
  }
  var args = arrayfromargs(arguments);
  try {
    target.apply(this, args);
  } catch (err) {
    debug("Unhandled exception in " + rawName + ": " + err);
    var requestId = fallbackRequestId(targetName, args);
    if (targetName === "api_session_clip_inspect") {
      sessionClipInspectionError(
        "internal_error",
        [],
        requestId
      );
      return;
    }
    ackWithRequest(
      "error",
      ["api_wrapper_internal_error", targetName],
      requestId
    );
  }
}

function clampMidiByte(value, fallback, contextName, label) {
  var n = Math.floor(Number(value));
  if (n >= 0 && n <= 127) {
    return n;
  }
  debug("Invalid MIDI " + label + " in " + contextName + ": " + value + " (using " + fallback + ")");
  return fallback;
}

function clampMidiChannel(value, fallback, contextName) {
  var ch = Math.floor(Number(value));
  if (ch >= 1 && ch <= 16) {
    return ch;
  }
  debug("Invalid MIDI channel in " + contextName + ": " + value + " (using " + fallback + ")");
  return fallback;
}

function midiCcStatusByte(channel) {
  // Control Change status byte is 0xB0 (176) + zero-based channel index.
  return 176 + (channel - 1);
}

function emitMidiCc(controller, value, channel, contextName) {
  var ctrl = clampMidiByte(controller, 64, contextName, "controller");
  var val = clampMidiByte(value, 0, contextName, "value");
  var ch = clampMidiChannel(channel, 1, contextName);
  var status = midiCcStatusByte(ch);
  outlet(2, status, ctrl, val);
  return { controller: ctrl, value: val, channel: ch, status: status };
}

function midi_cc(authToken, controller, value, channel, requestId) {
  if (!requireMutationAuth("midi_cc", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var result = emitMidiCc(controller, value, channel, "midi_cc");
  ackWithRequest("midi_cc", [result.controller, result.value, result.channel], requestId);
}

function cc64(authToken, value, channel, requestId) {
  if (!requireMutationAuth("cc64", authToken, requestId)) return;
  if (!ensureInitialized(requestId)) return;
  var result = emitMidiCc(64, value, channel, "cc64");
  ackWithRequest("cc64", [result.value, result.channel], requestId);
}

function tempo(authToken, bpm) {
  if (!requireMutationAuth("tempo", authToken)) return;
  if (!ensureInitialized()) return;
  var value = Number(bpm);
  if (!(value > 0)) {
    debug("Ignoring invalid tempo: " + bpm);
    return;
  }
  song.set("tempo", value);
  ack("ack", "tempo", value);
}

function sig_num(authToken, num) {
  if (!requireMutationAuth("sig_num", authToken)) return;
  if (!ensureInitialized()) return;
  var value = Math.floor(Number(num));
  if (!(value > 0)) {
    debug("Ignoring invalid signature numerator: " + num);
    return;
  }
  song.set("signature_numerator", value);
  ack("ack", "sig_num", value);
}

function sig_den(authToken, den) {
  if (!requireMutationAuth("sig_den", authToken)) return;
  if (!ensureInitialized()) return;
  var value = Math.floor(Number(den));
  if (!(value > 0)) {
    debug("Ignoring invalid signature denominator: " + den);
    return;
  }
  song.set("signature_denominator", value);
  ack("ack", "sig_den", value);
}

function create_midi_track(authToken) {
  if (!requireMutationAuth("create_midi_track", authToken)) return;
  if (!ensureInitialized()) return;
  song.call("create_midi_track", -1);
  ack("ack", "create_midi_track", -1);
}

function pad2(n) {
  return n < 10 ? "0" + String(n) : String(n);
}

function normalizePrefix(prefix, fallback) {
  if (prefix === undefined || prefix === null) return fallback;
  var text = String(prefix).trim();
  return text.length > 0 ? text : fallback;
}

function renameTrack(trackIndex, name) {
  try {
    var track = new LiveAPI(null, "live_set tracks " + trackIndex);
    track.set("name", name);
    return true;
  } catch (err) {
    debug("Failed to rename track " + trackIndex + " to '" + name + "': " + err);
    ack("ack", "error", "rename_track", trackIndex, name);
    return false;
  }
}

function create_audio_track(authToken) {
  if (!requireMutationAuth("create_audio_track", authToken)) return;
  if (!ensureInitialized()) return;
  song.call("create_audio_track", -1);
  ack("ack", "create_audio_track", -1);
}

function getTrackFlags(trackIndex) {
  var track = new LiveAPI(null, "live_set tracks " + trackIndex);
  var hasMidiInput = Number(getScalar(track, "has_midi_input"));
  var hasAudioInput = Number(getScalar(track, "has_audio_input"));
  return {
    track: track,
    hasMidiInput: hasMidiInput,
    hasAudioInput: hasAudioInput,
  };
}

function listTrackIndices(totalTracks, predicate, contextName) {
  var indices = [];
  for (var i = 0; i < totalTracks; i += 1) {
    try {
      var flags = getTrackFlags(i);
      if (predicate(flags)) {
        indices.push(i);
      }
    } catch (err) {
      debug("Failed to inspect track " + i + " in " + contextName + ": " + err);
      ack("ack", "error", "track_inspect_failed", contextName, i);
      return null;
    }
  }
  return indices;
}

function isAudioOnlyTrack(flags) {
  return flags.hasAudioInput === 1 && flags.hasMidiInput !== 1;
}

function isMidiTrack(flags) {
  return flags.hasMidiInput === 1;
}

function add_midi_tracks(authToken, count, name) {
  if (!requireMutationAuth("add_midi_tracks", authToken)) return;
  if (!ensureInitialized()) return;
  var targetCount = boundedInteger(count, 1, MAX_TRACKS_PER_COMMAND);
  if (targetCount === null) {
    debug("Ignoring out-of-range MIDI track count: " + count);
    ackWithRequest(
      "error",
      ["add_midi_tracks_count_out_of_range", count, MAX_TRACKS_PER_COMMAND]
    );
    return;
  }

  var initialTotal = getTotalTracksOrError("add_midi_tracks");
  if (initialTotal === 0) {
    return;
  }

  var trackName = normalizePrefix(name, "MIDI");
  var created = 0;

  for (var i = 0; i < targetCount; i += 1) {
    var before = getTotalTracksOrError("add_midi_tracks_before");
    if (before === 0) return;
    try {
      song.call("create_midi_track", -1);
    } catch (errCreate) {
      debug("Failed to create MIDI track " + i + ": " + errCreate);
      ack("ack", "error", "add_midi_tracks_create_failed", i);
      return;
    }
    var after = getTotalTracksOrError("add_midi_tracks_after");
    if (after === 0) return;

    var newIndex = after - 1;
    if (newIndex < before) {
      newIndex = before;
    }

    if (!renameTrack(newIndex, trackName)) return;
    created += 1;
    ack("ack", "midi_track_created", newIndex, trackName);
  }

  var finalTotal = getTotalTracksOrError("add_midi_tracks_final");
  if (finalTotal === 0) return;
  ack("ack", "add_midi_tracks", targetCount, trackName, created, finalTotal);
}

function getTotalTracksOrError(contextName, requestId) {
  var total = 0;
  try {
    total = song.getcount("tracks");
  } catch (err) {
    debug("Unable to read track count in " + contextName + ": " + err);
    ackWithRequest("error", ["track_count_failed", contextName], requestId);
    return 0;
  }
  if (total === 0) {
    debug("Track count is 0 in " + contextName + ". Device may not be attached to the Live set.");
    ackWithRequest("error", ["not_in_live_set", contextName], requestId);
  }
  return total;
}

function add_audio_tracks(authToken, count, prefix) {
  if (!requireMutationAuth("add_audio_tracks", authToken)) return;
  if (!ensureInitialized()) return;
  var targetCount = boundedInteger(count, 1, MAX_TRACKS_PER_COMMAND);
  if (targetCount === null) {
    debug("Ignoring out-of-range audio track count: " + count);
    ackWithRequest(
      "error",
      ["add_audio_tracks_count_out_of_range", count, MAX_TRACKS_PER_COMMAND]
    );
    return;
  }

  var initialTotal = getTotalTracksOrError("add_audio_tracks");
  if (initialTotal === 0) {
    return;
  }

  var namePrefix = normalizePrefix(prefix, "Audio");
  var created = 0;

  for (var i = 0; i < targetCount; i += 1) {
    var before = getTotalTracksOrError("add_audio_tracks_before");
    if (before === 0) return;
    try {
      song.call("create_audio_track", -1);
    } catch (errCreate) {
      debug("Failed to create audio track " + i + ": " + errCreate);
      ack("ack", "error", "add_audio_tracks_create_failed", i);
      return;
    }
    var after = getTotalTracksOrError("add_audio_tracks_after");
    if (after === 0) return;
    var newIndex = after - 1;
    if (newIndex < before) {
      newIndex = before;
    }

    var trackName = namePrefix + " " + pad2(i + 1);
    if (!renameTrack(newIndex, trackName)) return;
    created += 1;
    ack("ack", "audio_track_created", newIndex, trackName);
  }

  var finalTotal = getTotalTracksOrError("add_audio_tracks_final");
  if (finalTotal === 0) return;
  ack("ack", "add_audio_tracks", targetCount, namePrefix, created, finalTotal);
}

function delete_midi_tracks(authToken, count) {
  if (!requireMutationAuth("delete_midi_tracks", authToken)) return;
  if (!ensureInitialized()) return;
  var targetCount = boundedInteger(count, 1, MAX_TRACKS_PER_COMMAND);
  if (targetCount === null) {
    debug("Ignoring out-of-range MIDI delete count: " + count);
    ackWithRequest(
      "error",
      ["delete_midi_tracks_count_out_of_range", count, MAX_TRACKS_PER_COMMAND]
    );
    return;
  }

  var totalTracks = getTotalTracksOrError("delete_midi_tracks");
  if (totalTracks === 0) {
    return;
  }

  var midiIndices = listTrackIndices(totalTracks, isMidiTrack, "delete_midi_tracks");
  if (midiIndices === null) return;
  // Preserve track 0 as a stable default "do not delete" track.
  var deletableMidiIndices = midiIndices.filter(function (i) {
    return i > 0;
  });

  if (deletableMidiIndices.length === 0) {
    debug("No deletable MIDI tracks found (track 0 is protected).");
    ack("ack", "error", "no_midi_tracks");
    return;
  }

  var deleteIndices = deletableMidiIndices.slice(-targetCount).sort(function (a, b) {
    return b - a;
  });

  var deleted = 0;
  for (var i = 0; i < deleteIndices.length; i += 1) {
    var index = deleteIndices[i];
    try {
      song.call("delete_track", index);
      deleted += 1;
      ack("ack", "midi_track_deleted", index);
    } catch (err) {
      debug("Failed to delete MIDI track " + index + ": " + err);
      ack("ack", "error", "midi_track_delete_failed", index);
      return;
    }
  }

  var finalTotal = getTotalTracksOrError("delete_midi_tracks_final");
  if (finalTotal === 0) return;
  ack("ack", "delete_midi_tracks", targetCount, deleted, finalTotal);
}

function rename_track(authToken, trackIndex, name) {
  if (!requireMutationAuth("rename_track", authToken)) return;
  if (!ensureInitialized()) return;
  var index = Math.floor(Number(trackIndex));
  if (!(index >= 0)) {
    ack("ack", "error", "rename_track_invalid_index", trackIndex);
    return;
  }

  var totalTracks = getTotalTracksOrError("rename_track");
  if (totalTracks === 0) {
    return;
  }

  if (index >= totalTracks) {
    ack("ack", "error", "rename_track_index_out_of_range", index, totalTracks);
    return;
  }

  var trackName = normalizePrefix(name, "Track " + index);
  if (renameTrack(index, trackName)) {
    ack("ack", "track_renamed", index, trackName);
  }
}

function getTrackOrError(trackIndex, contextName) {
  var index = Math.floor(Number(trackIndex));
  if (!(index >= 0)) {
    ack("ack", "error", contextName + "_invalid_index", trackIndex);
    return null;
  }

  var totalTracks = getTotalTracksOrError(contextName);
  if (totalTracks === 0) {
    return null;
  }

  if (index >= totalTracks) {
    ack("ack", "error", contextName + "_index_out_of_range", index, totalTracks);
    return null;
  }

  try {
    return new LiveAPI(null, "live_set tracks " + index);
  } catch (err) {
    debug("Unable to access track " + index + " in " + contextName + ": " + err);
    ack("ack", "error", contextName + "_track_access_failed", index);
    return null;
  }
}

function parseNotesJson(notesJson, contextName) {
  var raw = notesJson;
  if (raw === undefined || raw === null) {
    ack("ack", "error", contextName + "_missing_notes");
    return null;
  }

  var text = String(raw);
  try {
    var parsed = JSON.parse(text);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    if (parsed && Array.isArray(parsed.notes)) {
      return parsed.notes;
    }
    ack("ack", "error", contextName + "_notes_not_array");
    return null;
  } catch (err) {
    debug("Failed to parse notes JSON in " + contextName + ": " + err);
    ack("ack", "error", contextName + "_notes_json_parse_failed");
    return null;
  }
}

function copyOptionalNoteNumber(note, normalized, fieldName, minValue, maxValue, index, contextName, integerValue, requestId) {
  var raw = note[fieldName];
  if (raw === undefined || raw === null || String(raw).length === 0) {
    return true;
  }

  var value = Number(raw);
  if (!(isFinite(value) && value >= minValue && value <= maxValue)) {
    ackWithRequest("error", [contextName + "_invalid_" + fieldName, index, raw], requestId);
    return false;
  }

  normalized[fieldName] = integerValue ? Math.floor(value) : value;
  return true;
}

function normalizeNote(note, index, contextName, requestId) {
  var pitch = Math.floor(Number(note.pitch));
  var startTime = Number(note.start_time);
  var duration = Number(note.duration);
  var velocity = 100;
  if (note.velocity !== undefined && note.velocity !== null && String(note.velocity).length > 0) {
    velocity = Math.floor(Number(note.velocity));
  }
  var mute = Number(note.mute) ? 1 : 0;

  if (!(pitch >= 0 && pitch <= 127)) {
    ackWithRequest("error", [contextName + "_invalid_pitch", index, note.pitch], requestId);
    return null;
  }
  if (!(startTime >= 0)) {
    ackWithRequest("error", [contextName + "_invalid_start_time", index, note.start_time], requestId);
    return null;
  }
  if (!(duration > 0)) {
    ackWithRequest("error", [contextName + "_invalid_duration", index, note.duration], requestId);
    return null;
  }
  if (!(velocity >= 0 && velocity <= 127)) {
    velocity = 100;
  }

  var normalized = {
    pitch: pitch,
    start_time: startTime,
    duration: duration,
    velocity: velocity,
    mute: mute,
  };

  if (!copyOptionalNoteNumber(note, normalized, "probability", 0, 1, index, contextName, false, requestId)) {
    return null;
  }
  if (!copyOptionalNoteNumber(note, normalized, "velocity_deviation", -127, 127, index, contextName, false, requestId)) {
    return null;
  }
  if (!copyOptionalNoteNumber(note, normalized, "release_velocity", 0, 127, index, contextName, true, requestId)) {
    return null;
  }

  return normalized;
}

function buildNotesDict(notes, contextName, requestId) {
  var normalized = [];
  for (var i = 0; i < notes.length; i += 1) {
    var norm = normalizeNote(notes[i], i, contextName, requestId);
    if (!norm) {
      return null;
    }
    normalized.push(norm);
  }

  var notesData = { notes: normalized };
  // In Max JS, arrays of dictionaries often need a wrapper key to be parsed
  // into a Dict that LiveAPI accepts as a dictionary argument.
  var wrapperName = "live_bridge_notes_wrapper_" + new Date().getTime();
  var wrapper = null;
  var notesDict = null;
  try {
    wrapper = new Dict(wrapperName);
    wrapper.setparse("wrapper", JSON.stringify(notesData));
    notesDict = wrapper.get("wrapper");
    if (!notesDict) {
      throw new Error("Dict wrapper did not return a notes dictionary");
    }
  } catch (err) {
    debug("Failed to build notes Dict for " + contextName + ": " + err);
    ackWithRequest("error", [contextName + "_notes_dict_build_failed"], requestId);
    clearBuiltPayload({ wrapper: wrapper });
    return null;
  }
  return { wrapper: wrapper, dict: notesDict, notes: normalized };
}

function buildGenericDict(payload, contextName, requestId) {
  var wrapperName = "live_bridge_dict_wrapper_" + new Date().getTime();
  var wrapper = null;
  var parsedDict = null;
  try {
    wrapper = new Dict(wrapperName);
    wrapper.setparse("wrapper", JSON.stringify(payload));
    parsedDict = wrapper.get("wrapper");
    if (!parsedDict) {
      throw new Error("Dict wrapper did not return a dictionary");
    }
  } catch (err) {
    debug("Failed to build Dict for " + contextName + ": " + err);
    ackWithRequest("error", [contextName + "_dict_build_failed"], requestId);
    clearBuiltPayload({ wrapper: wrapper });
    return null;
  }
  return { wrapper: wrapper, dict: parsedDict };
}

function set_session_clip_notes(authToken, trackIndex, slotIndex, lengthBeats, notesJson, clipName) {
  if (!requireMutationAuth("set_session_clip_notes", authToken)) return;
  if (!ensureInitialized()) return;
  var startedMs = new Date().getTime();

  var contextName = "set_session_clip_notes";
  var track = getTrackOrError(trackIndex, contextName);
  if (!track) return;

  var hasMidiInput = Number(getScalar(track, "has_midi_input"));
  if (hasMidiInput !== 1) {
    ack("ack", "error", contextName + "_track_not_midi", trackIndex);
    return;
  }

  var slot = Math.floor(Number(slotIndex));
  if (!(slot >= 0)) {
    ack("ack", "error", contextName + "_invalid_slot_index", slotIndex);
    return;
  }

  var length = Number(lengthBeats);
  if (!(length > 0)) {
    ack("ack", "error", contextName + "_invalid_length", lengthBeats);
    return;
  }

  var notes = parseNotesJson(notesJson, contextName);
  if (!notes || notes.length === 0) {
    ack("ack", "error", contextName + "_no_notes");
    return;
  }

  var built = buildNotesDict(notes, contextName);
  if (!built) {
    return;
  }

  var slotPath = "live_set tracks " + Math.floor(Number(trackIndex)) + " clip_slots " + slot;
  var clipSlot = null;
  try {
    clipSlot = new LiveAPI(null, slotPath);
  } catch (err) {
    debug("Unable to access clip slot at " + slotPath + ": " + err);
    ack("ack", "error", contextName + "_clip_slot_access_failed", trackIndex, slot);
    clearBuiltPayload(built);
    return;
  }

  // Ensure the slot is empty before creating a clip.
  try {
    clipSlot.call("delete_clip");
  } catch (err) {
    // It's fine if there was no clip to delete.
  }

  try {
    clipSlot.call("create_clip", length);
  } catch (err) {
    debug("Failed to create clip at " + slotPath + " length=" + length + ": " + err);
    ack("ack", "error", contextName + "_create_clip_failed", trackIndex, slot, length);
    clearBuiltPayload(built);
    return;
  }

  var clipPath = slotPath + " clip";
  var clip = null;
  try {
    clip = new LiveAPI(null, clipPath);
  } catch (err) {
    debug("Unable to access clip at " + clipPath + ": " + err);
    ack("ack", "error", contextName + "_clip_access_failed", trackIndex, slot);
    clearBuiltPayload(built);
    return;
  }

  try {
    clip.set("loop_start", 0);
    clip.set("loop_end", length);
  } catch (err) {
    debug("Unable to set loop properties on clip at " + clipPath + ": " + err);
  }

  var nameText = normalizePrefix(clipName, "");
  if (nameText.length > 0) {
    try {
      clip.set("name", nameText);
    } catch (err) {
      debug("Failed to name clip '" + nameText + "': " + err);
    }
  }

  try {
    clip.call("deselect_all_notes");
  } catch (err) {
    // Best effort; not required for add_new_notes.
  }

  var noteIds = [];
  try {
    noteIds = clip.call("add_new_notes", built.dict);
    if (!Array.isArray(noteIds) || noteIds.length === 0) {
      noteIds = clip.call("add_new_notes", built.dict.name);
    }
  } catch (err) {
    debug("Failed to add notes to clip at " + clipPath + ": " + err);
    ack("ack", "error", contextName + "_add_notes_failed");
    return;
  } finally {
    clearBuiltPayload(built);
  }

  var noteIdCount = Array.isArray(noteIds) ? noteIds.length : 0;
  ack(
    "ack",
    "set_session_clip_notes",
    Math.floor(Number(trackIndex)),
    slot,
    length,
    built.notes.length,
    noteIdCount,
    nameText
  );
  debug(
    "set_session_clip_notes elapsed_ms=" + (new Date().getTime() - startedMs)
  );
}

function append_session_clip_notes(authToken, trackIndex, slotIndex, notesJson) {
  if (!requireMutationAuth("append_session_clip_notes", authToken)) return;
  if (!ensureInitialized()) return;
  var startedMs = new Date().getTime();

  var contextName = "append_session_clip_notes";
  var track = getTrackOrError(trackIndex, contextName);
  if (!track) return;

  var hasMidiInput = Number(getScalar(track, "has_midi_input"));
  if (hasMidiInput !== 1) {
    ack("ack", "error", contextName + "_track_not_midi", trackIndex);
    return;
  }

  var notes = parseNotesJson(notesJson, contextName);
  if (!notes || notes.length === 0) {
    ack("ack", "error", contextName + "_no_notes");
    return;
  }

  var built = buildNotesDict(notes, contextName);
  if (!built) {
    return;
  }

  var clip = getClipFromSlotOrError(trackIndex, slotIndex, contextName);
  if (!clip) {
    clearBuiltPayload(built);
    return;
  }

  var noteIds = [];
  try {
    noteIds = clip.call("add_new_notes", built.dict);
    if (!Array.isArray(noteIds) || noteIds.length === 0) {
      noteIds = clip.call("add_new_notes", built.dict.name);
    }
  } catch (err) {
    debug("Failed to append notes to clip: " + err);
    ack("ack", "error", contextName + "_add_notes_failed");
    return;
  } finally {
    clearBuiltPayload(built);
  }

  var noteIdCount = Array.isArray(noteIds) ? noteIds.length : 0;
  ack(
    "ack",
    "append_session_clip_notes",
    Math.floor(Number(trackIndex)),
    Math.floor(Number(slotIndex)),
    built.notes.length,
    noteIdCount
  );
  debug(
    "append_session_clip_notes elapsed_ms=" + (new Date().getTime() - startedMs)
  );
}

function getClipSlotOrError(trackIndex, slotIndex, contextName) {
  var slot = Math.floor(Number(slotIndex));
  if (!(slot >= 0)) {
    ack("ack", "error", contextName + "_invalid_slot_index", slotIndex);
    return null;
  }

  var slotPath = "live_set tracks " + Math.floor(Number(trackIndex)) + " clip_slots " + slot;
  try {
    return new LiveAPI(null, slotPath);
  } catch (err) {
    debug("Unable to access clip slot at " + slotPath + ": " + err);
    ack("ack", "error", contextName + "_clip_slot_access_failed", trackIndex, slot);
    return null;
  }
}

function getClipFromSlotOrError(trackIndex, slotIndex, contextName) {
  var clipSlot = getClipSlotOrError(trackIndex, slotIndex, contextName);
  if (!clipSlot) return null;

  var hasClip = Number(getScalar(clipSlot, "has_clip"));
  if (hasClip !== 1) {
    ack("ack", "error", contextName + "_no_clip", trackIndex, slotIndex);
    return null;
  }

  var clipPath =
    "live_set tracks " +
    Math.floor(Number(trackIndex)) +
    " clip_slots " +
    Math.floor(Number(slotIndex)) +
    " clip";
  try {
    return new LiveAPI(null, clipPath);
  } catch (err) {
    debug("Unable to access clip at " + clipPath + ": " + err);
    ack("ack", "error", contextName + "_clip_access_failed", trackIndex, slotIndex);
    return null;
  }
}

function inspect_session_clip_notes(trackIndex, slotIndex) {
  if (!ensureInitialized()) return;

  var contextName = "inspect_session_clip_notes";
  var track = getTrackOrError(trackIndex, contextName);
  if (!track) return;

  var hasMidiInput = Number(getScalar(track, "has_midi_input"));
  if (hasMidiInput !== 1) {
    ack("ack", "error", contextName + "_track_not_midi", trackIndex);
    return;
  }

  var clip = getClipFromSlotOrError(trackIndex, slotIndex, contextName);
  if (!clip) return;

  var noteCount = 0;
  var minPitch = -1;
  var maxPitch = -1;
  var clipLength = 0;
  var rawResult = "";

  try {
    clipLength = Number(getScalar(clip, "length"));
  } catch (err) {
    clipLength = 0;
  }

  try {
    var result = clip.call("get_all_notes_extended");
    rawResult = Array.isArray(result) ? result.join(" ") : String(result);
    var parsed = JSON.parse(rawResult || "{}");
    var notes = Array.isArray(parsed.notes) ? parsed.notes : [];
    noteCount = notes.length;
    if (noteCount > 0) {
      minPitch = notes[0].pitch;
      maxPitch = notes[0].pitch;
      for (var i = 1; i < notes.length; i += 1) {
        var pitch = notes[i].pitch;
        if (pitch < minPitch) minPitch = pitch;
        if (pitch > maxPitch) maxPitch = pitch;
      }
    }
  } catch (err) {
    debug("Failed to inspect notes: " + err);
    ack("ack", "error", contextName + "_inspect_failed");
    return;
  }

  ack(
    "ack",
    "inspect_session_clip_notes",
    Math.floor(Number(trackIndex)),
    Math.floor(Number(slotIndex)),
    noteCount,
    minPitch,
    maxPitch,
    clipLength,
    rawResult
  );
}

function countMidiTracks(totalTracks, contextName, requestId) {
  var midiCount = 0;
  for (var i = 0; i < totalTracks; i += 1) {
    try {
      var track = new LiveAPI(null, "live_set tracks " + i);
      var hasMidiInput = Number(getScalar(track, "has_midi_input"));
      if (hasMidiInput === 1) {
        midiCount += 1;
      }
    } catch (err) {
      var midiContext = String(contextName || "count_midi_tracks");
      debug("Failed to inspect MIDI track " + i + " in " + midiContext + ": " + err);
      ackWithRequest("error", ["count_midi_tracks_failed", midiContext, i], requestId);
      return null;
    }
  }
  return midiCount;
}

function countAudioTracks(totalTracks, contextName, requestId) {
  var audioCount = 0;
  for (var i = 0; i < totalTracks; i += 1) {
    try {
      var track = new LiveAPI(null, "live_set tracks " + i);
      var hasAudioInput = Number(getScalar(track, "has_audio_input"));
      if (hasAudioInput === 1) {
        audioCount += 1;
      }
    } catch (err) {
      var audioContext = String(contextName || "count_audio_tracks");
      debug("Failed to inspect audio track " + i + " in " + audioContext + ": " + err);
      ackWithRequest("error", ["count_audio_tracks_failed", audioContext, i], requestId);
      return null;
    }
  }
  return audioCount;
}

function delete_audio_tracks(authToken, count) {
  if (!requireMutationAuth("delete_audio_tracks", authToken)) return;
  if (!ensureInitialized()) return;
  var targetCount = boundedInteger(count, 1, MAX_TRACKS_PER_COMMAND);
  if (targetCount === null) {
    debug("Ignoring out-of-range audio delete count: " + count);
    ackWithRequest(
      "error",
      ["delete_audio_tracks_count_out_of_range", count, MAX_TRACKS_PER_COMMAND]
    );
    return;
  }

  var totalTracks = getTotalTracksOrError("delete_audio_tracks");
  if (totalTracks === 0) {
    return;
  }

  var audioIndices = listTrackIndices(totalTracks, isAudioOnlyTrack, "delete_audio_tracks");
  if (audioIndices === null) return;
  if (audioIndices.length === 0) {
    debug("No audio tracks found to delete.");
    ack("ack", "error", "no_audio_tracks");
    return;
  }

  var toDelete = audioIndices.slice(-targetCount).reverse();
  var deleted = 0;
  for (var i = 0; i < toDelete.length; i += 1) {
    var trackIndex = toDelete[i];
    try {
      song.call("delete_track", trackIndex);
      deleted += 1;
      ack("ack", "audio_track_deleted", trackIndex);
    } catch (err) {
      debug("Failed to delete audio track " + trackIndex + ": " + err);
      ack("ack", "error", "delete_audio_track_failed", trackIndex);
      return;
    }
  }

  var finalTotal = getTotalTracksOrError("delete_audio_tracks_final");
  if (finalTotal === 0) return;
  ack("ack", "delete_audio_tracks", targetCount, deleted, finalTotal);
}

function status() {
  if (!ensureInitialized()) return;
  var totalTracks = getTotalTracksOrError("status");
  if (totalTracks === 0) return;
  var returnTracks = 0;
  try {
    returnTracks = song.getcount("return_tracks");
  } catch (err) {
    debug("Unable to read return track count: " + err);
  }
  var midiTracks = countMidiTracks(totalTracks, "status");
  if (midiTracks === null) return;
  var audioTracks = countAudioTracks(totalTracks, "status");
  if (audioTracks === null) return;
  var id = song ? Number(song.id) : 0;
  ack("ack", "status", totalTracks, midiTracks, audioTracks, returnTracks, song.path, id);
}

function ensure_midi_tracks(authToken, targetCount) {
  if (!requireMutationAuth("ensure_midi_tracks", authToken)) return;
  if (!ensureInitialized()) return;
  var target = boundedInteger(targetCount, 0, MAX_TRACK_TARGET);
  if (target === null) {
    debug("Ignoring out-of-range target track count: " + targetCount);
    ackWithRequest(
      "error",
      ["ensure_midi_tracks_target_out_of_range", targetCount, MAX_TRACK_TARGET]
    );
    return;
  }

  var totalTracks = getTotalTracksOrError("ensure_midi_tracks");
  if (totalTracks === 0) {
    return;
  }

  var currentMidiTracks = countMidiTracks(totalTracks, "ensure_midi_tracks");
  if (currentMidiTracks === null) return;
  var missing = target - currentMidiTracks;
  if (missing <= 0) {
    ack("ack", "ensure_midi_tracks", target, currentMidiTracks, 0, totalTracks);
    return;
  }
  if (missing > MAX_TRACKS_PER_COMMAND) {
    ackWithRequest(
      "error",
      [
        "ensure_midi_tracks_batch_too_large",
        missing,
        MAX_TRACKS_PER_COMMAND,
      ]
    );
    return;
  }

  for (var i = 0; i < missing; i += 1) {
    song.call("create_midi_track", -1);
  }
  ack("ack", "ensure_midi_tracks", target, currentMidiTracks, missing, totalTracks);
}

function emitAck(args) {
  // Emit OSC-friendly messages via udpsend. We use a leading slash address.
  // Example: /ack tempo 120
  if (args.length === 0) return;

  var address = "/" + String(args[0]);
  var rest = args.slice(1);
  var message = [0, address].concat(rest);
  outlet.apply(this, message);
}

function ack() {
  var args = Array.prototype.slice.call(arguments);
  if (args[0] === "ack" && args[1] === "error") {
    args.push(ERROR_CORRELATION_MARKER);
    args.push("req:");
  }
  emitAck(args);
}
