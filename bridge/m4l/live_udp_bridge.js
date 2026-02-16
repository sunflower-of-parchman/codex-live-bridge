// Ableton Live UDP bridge logic for Max for Live.
// This file is intended to be loaded via: [js live_udp_bridge.js]

autowatch = 1;
inlets = 1;
outlets = 3; // 0 -> UDP ack/debug, 1 -> console/debug, 2 -> MIDI out

var song = null;
var initialized = false;

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
  if (hasRequestId(requestId)) {
    payload.push(String(requestId));
  }
  ack.apply(this, payload);
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

function resolveApiOrError(path, contextName, requestId) {
  var pathText = path === undefined || path === null ? "" : String(path).trim();
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

function ensureInitialized() {
  var currentId = song ? Number(song.id) : 0;
  if (initialized && song && currentId > 0) {
    return true;
  }
  init();
  currentId = song ? Number(song.id) : 0;
  if (initialized && song && currentId > 0) {
    return true;
  }
  debug("LiveAPI not initialized yet or not attached to live_set.");
  ack("ack", "error", "not_initialized");
  return false;
}

function init() {
  try {
    song = new LiveAPI(null, "live_set");
    var id = song ? Number(song.id) : 0;
    if (!(id > 0)) {
      initialized = false;
      debug("LiveAPI attached to live_set but id is invalid: " + id);
      ack("ack", "error", "not_in_live_set", id);
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
  if (!ensureInitialized()) return;
  ackWithRequest("pong", [], requestId);
}

function api_get(path, property, requestId) {
  if (!ensureInitialized()) return;
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

function api_set(path, property, valueJson, requestId) {
  if (!ensureInitialized()) return;
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

function api_call(path, method, argsJson, requestId) {
  if (!ensureInitialized()) return;
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
    builtPayload = buildNotesDict(notesList, contextName + "_add_new_notes");
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
    builtPayload = buildGenericDict(payload, contextName + "_" + methodName);
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
    if (builtPayload) {
      try {
        if (builtPayload.wrapper) {
          builtPayload.wrapper.clear();
        } else if (builtPayload.dict) {
          builtPayload.dict.clear();
        }
      } catch (cleanupErr) {
        // Best-effort cleanup.
      }
    }
  }

  var resultJson = liveApiValueToJson(result, contextName + "_" + methodName);
  debug(
    "api_call " + methodName + " elapsed_ms=" + (new Date().getTime() - startedMs)
  );
  ackWithRequest("api_call", [api.path, methodName, resultJson], requestId);
}

function api_children(path, childName, requestId) {
  if (!ensureInitialized()) return;
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
  for (var i = 0; i < count; i += 1) {
    var childPath = api.path + " " + childProp + " " + i;
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
  ackWithRequest("api_children", [api.path, childProp, childrenJson], requestId);
}

function api_describe(path, requestId) {
  if (!ensureInitialized()) return;
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

function midi_cc(controller, value, channel, requestId) {
  if (!ensureInitialized()) return;
  var result = emitMidiCc(controller, value, channel, "midi_cc");
  ackWithRequest("midi_cc", [result.controller, result.value, result.channel], requestId);
}

function cc64(value, channel, requestId) {
  if (!ensureInitialized()) return;
  var result = emitMidiCc(64, value, channel, "cc64");
  ackWithRequest("cc64", [result.value, result.channel], requestId);
}

function tempo(bpm) {
  if (!ensureInitialized()) return;
  var value = Number(bpm);
  if (!(value > 0)) {
    debug("Ignoring invalid tempo: " + bpm);
    return;
  }
  song.set("tempo", value);
  ack("ack", "tempo", value);
}

function sig_num(num) {
  if (!ensureInitialized()) return;
  var value = Math.floor(Number(num));
  if (!(value > 0)) {
    debug("Ignoring invalid signature numerator: " + num);
    return;
  }
  song.set("signature_numerator", value);
  ack("ack", "sig_num", value);
}

function sig_den(den) {
  if (!ensureInitialized()) return;
  var value = Math.floor(Number(den));
  if (!(value > 0)) {
    debug("Ignoring invalid signature denominator: " + den);
    return;
  }
  song.set("signature_denominator", value);
  ack("ack", "sig_den", value);
}

function create_midi_track() {
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

function create_audio_track() {
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
      break;
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

function add_midi_tracks(count, name) {
  if (!ensureInitialized()) return;
  var targetCount = Math.floor(Number(count));
  if (!(targetCount > 0)) {
    debug("Ignoring invalid MIDI track count: " + count);
    ack("ack", "error", "add_midi_tracks_invalid_count", count);
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
    if (before === 0) break;
    song.call("create_midi_track", -1);
    var after = getTotalTracksOrError("add_midi_tracks_after");
    if (after === 0) break;

    var newIndex = after - 1;
    if (newIndex < before) {
      newIndex = before;
    }

    renameTrack(newIndex, trackName);
    created += 1;
    ack("ack", "midi_track_created", newIndex, trackName);
  }

  var finalTotal = getTotalTracksOrError("add_midi_tracks_final");
  ack("ack", "add_midi_tracks", targetCount, trackName, created, finalTotal);
}

function getTotalTracksOrError(contextName) {
  var total = 0;
  try {
    total = song.getcount("tracks");
  } catch (err) {
    debug("Unable to read track count in " + contextName + ": " + err);
    ack("ack", "error", "track_count_failed", contextName);
    return 0;
  }
  if (total === 0) {
    debug("Track count is 0 in " + contextName + ". Device may not be attached to the Live set.");
    ack("ack", "error", "not_in_live_set", contextName);
  }
  return total;
}

function add_audio_tracks(count, prefix) {
  if (!ensureInitialized()) return;
  var targetCount = Math.floor(Number(count));
  if (!(targetCount > 0)) {
    debug("Ignoring invalid audio track count: " + count);
    ack("ack", "error", "add_audio_tracks_invalid_count", count);
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
    if (before === 0) break;
    song.call("create_audio_track", -1);
    var after = getTotalTracksOrError("add_audio_tracks_after");
    if (after === 0) break;
    var newIndex = after - 1;
    if (newIndex < before) {
      newIndex = before;
    }

    var trackName = namePrefix + " " + pad2(i + 1);
    renameTrack(newIndex, trackName);
    created += 1;
    ack("ack", "audio_track_created", newIndex, trackName);
  }

  var finalTotal = getTotalTracksOrError("add_audio_tracks_final");
  ack("ack", "add_audio_tracks", targetCount, namePrefix, created, finalTotal);
}

function delete_midi_tracks(count) {
  if (!ensureInitialized()) return;
  var targetCount = Math.floor(Number(count));
  if (!(targetCount > 0)) {
    debug("Ignoring invalid MIDI delete count: " + count);
    ack("ack", "error", "delete_midi_tracks_invalid_count", count);
    return;
  }

  var totalTracks = getTotalTracksOrError("delete_midi_tracks");
  if (totalTracks === 0) {
    return;
  }

  var midiIndices = listTrackIndices(totalTracks, isMidiTrack, "delete_midi_tracks");
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
      break;
    }
  }

  var finalTotal = getTotalTracksOrError("delete_midi_tracks_final");
  ack("ack", "delete_midi_tracks", targetCount, deleted, finalTotal);
}

function rename_track(trackIndex, name) {
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

function normalizeNote(note, index, contextName) {
  var pitch = Math.floor(Number(note.pitch));
  var startTime = Number(note.start_time);
  var duration = Number(note.duration);
  var velocity = Math.floor(Number(note.velocity));
  var mute = Number(note.mute) ? 1 : 0;

  if (!(pitch >= 0 && pitch <= 127)) {
    ack("ack", "error", contextName + "_invalid_pitch", index, note.pitch);
    return null;
  }
  if (!(startTime >= 0)) {
    ack("ack", "error", contextName + "_invalid_start_time", index, note.start_time);
    return null;
  }
  if (!(duration > 0)) {
    ack("ack", "error", contextName + "_invalid_duration", index, note.duration);
    return null;
  }
  if (!(velocity > 0 && velocity <= 127)) {
    velocity = 100;
  }

  return {
    pitch: pitch,
    start_time: startTime,
    duration: duration,
    velocity: velocity,
    mute: mute,
  };
}

function buildNotesDict(notes, contextName) {
  var normalized = [];
  for (var i = 0; i < notes.length; i += 1) {
    var norm = normalizeNote(notes[i], i, contextName);
    if (!norm) {
      return null;
    }
    normalized.push(norm);
  }

  var notesData = { notes: normalized };
  // In Max JS, arrays of dictionaries often need a wrapper key to be parsed
  // into a Dict that LiveAPI accepts as a dictionary argument.
  var wrapperName = "live_bridge_notes_wrapper_" + new Date().getTime();
  var wrapper = new Dict(wrapperName);
  wrapper.setparse("wrapper", JSON.stringify(notesData));
  var notesDict = wrapper.get("wrapper");
  if (!notesDict) {
    ack("ack", "error", contextName + "_notes_dict_build_failed");
    return null;
  }
  return { wrapper: wrapper, dict: notesDict, notes: normalized };
}

function buildGenericDict(payload, contextName) {
  var wrapperName = "live_bridge_dict_wrapper_" + new Date().getTime();
  var wrapper = new Dict(wrapperName);
  try {
    wrapper.setparse("wrapper", JSON.stringify(payload));
  } catch (err) {
    debug("Failed to build Dict for " + contextName + ": " + err);
    ack("ack", "error", contextName + "_dict_build_failed");
    return null;
  }
  var parsedDict = wrapper.get("wrapper");
  if (!parsedDict) {
    ack("ack", "error", contextName + "_dict_build_failed");
    return null;
  }
  return { wrapper: wrapper, dict: parsedDict };
}

function set_session_clip_notes(trackIndex, slotIndex, lengthBeats, notesJson, clipName) {
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
    return;
  }

  var clipPath = slotPath + " clip";
  var clip = null;
  try {
    clip = new LiveAPI(null, clipPath);
  } catch (err) {
    debug("Unable to access clip at " + clipPath + ": " + err);
    ack("ack", "error", contextName + "_clip_access_failed", trackIndex, slot);
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
    try {
      if (built.wrapper) {
        built.wrapper.clear();
      } else if (built.dict) {
        built.dict.clear();
      }
    } catch (err) {
      // Best-effort cleanup.
    }
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

function append_session_clip_notes(trackIndex, slotIndex, notesJson) {
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
  if (!clip) return;

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
    try {
      if (built.wrapper) {
        built.wrapper.clear();
      } else if (built.dict) {
        built.dict.clear();
      }
    } catch (err) {
      // Best-effort cleanup.
    }
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

function countMidiTracks(totalTracks) {
  var midiCount = 0;
  for (var i = 0; i < totalTracks; i += 1) {
    var track = new LiveAPI(null, "live_set tracks " + i);
    var hasMidiInput = Number(getScalar(track, "has_midi_input"));
    if (hasMidiInput === 1) {
      midiCount += 1;
    }
  }
  return midiCount;
}

function countAudioTracks(totalTracks) {
  var audioCount = 0;
  for (var i = 0; i < totalTracks; i += 1) {
    try {
      var track = new LiveAPI(null, "live_set tracks " + i);
      var hasAudioInput = Number(getScalar(track, "has_audio_input"));
      if (hasAudioInput === 1) {
        audioCount += 1;
      }
    } catch (err) {
      debug("Failed to inspect audio track " + i + ": " + err);
      ack("ack", "error", "count_audio_tracks_failed", i);
      return audioCount;
    }
  }
  return audioCount;
}

function delete_audio_tracks(count) {
  if (!ensureInitialized()) return;
  var targetCount = Math.floor(Number(count));
  if (!(targetCount > 0)) {
    debug("Ignoring invalid delete audio track count: " + count);
    ack("ack", "error", "delete_audio_tracks_invalid_count", count);
    return;
  }

  var totalTracks = getTotalTracksOrError("delete_audio_tracks");
  if (totalTracks === 0) {
    return;
  }

  var audioIndices = listTrackIndices(totalTracks, isAudioOnlyTrack, "delete_audio_tracks");
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
    }
  }

  var finalTotal = getTotalTracksOrError("delete_audio_tracks_final");
  ack("ack", "delete_audio_tracks", targetCount, deleted, finalTotal);
}

function status() {
  if (!ensureInitialized()) return;
  var totalTracks = getTotalTracksOrError("status");
  var returnTracks = 0;
  try {
    returnTracks = song.getcount("return_tracks");
  } catch (err) {
    debug("Unable to read return track count: " + err);
  }
  var midiTracks = totalTracks > 0 ? countMidiTracks(totalTracks) : 0;
  var audioTracks = totalTracks > 0 ? countAudioTracks(totalTracks) : 0;
  var id = song ? Number(song.id) : 0;
  ack("ack", "status", totalTracks, midiTracks, audioTracks, returnTracks, song.path, id);
}

function ensure_midi_tracks(targetCount) {
  if (!ensureInitialized()) return;
  var target = Math.floor(Number(targetCount));
  if (!(target >= 0)) {
    debug("Ignoring invalid target track count: " + targetCount);
    return;
  }

  var totalTracks = getTotalTracksOrError("ensure_midi_tracks");
  if (totalTracks === 0) {
    return;
  }

  var currentMidiTracks = countMidiTracks(totalTracks);
  var missing = target - currentMidiTracks;
  if (missing <= 0) {
    ack("ack", "ensure_midi_tracks", target, currentMidiTracks, 0, totalTracks);
    return;
  }

  for (var i = 0; i < missing; i += 1) {
    song.call("create_midi_track", -1);
  }
  ack("ack", "ensure_midi_tracks", target, currentMidiTracks, missing, totalTracks);
}

function ack() {
  // Emit OSC-friendly messages via udpsend. We use a leading slash address.
  // Example: /ack tempo 120
  var args = Array.prototype.slice.call(arguments);
  if (args.length === 0) return;

  var address = "/" + String(args[0]);
  var rest = args.slice(1);
  var message = [0, address].concat(rest);
  outlet.apply(this, message);
}
