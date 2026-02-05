autowatch = 1;
inlets = 1;
outlets = 1;

function post_json(obj) {
  var raw = JSON.stringify(obj);
  var bytes = [];
  for (var i = 0; i < raw.length; i++) {
    bytes.push(raw.charCodeAt(i) & 255);
  }
  outlet(0, bytes);
}

function error_json(id, message) {
  post_json({ ok: false, id: id || null, error: message });
}

function success_json(id, result) {
  post_json({ ok: true, id: id || null, result: result || {} });
}

function parse_message(args) {
  if (!args || !args.length) {
    throw new Error("No command payload provided.");
  }
  if (args.length === 1 && typeof args[0] === "string" && args[0].charAt(0) === "{") {
    return JSON.parse(args[0]);
  }
  var raw = args.join(" ");
  return JSON.parse(raw);
}

function anything() {
  var args;
  var payload;
  try {
    if (messagename === "rawbytes") {
      args = arrayfromargs(arguments);
      payload = parse_rawbytes(args);
      handle_command(payload);
      return;
    }
    args = arrayfromargs(messagename, arguments);
    payload = parse_message(args);
    handle_command(payload);
  } catch (err) {
    error_json(null, String(err));
  }
}

function list() {
  var args = arrayfromargs(arguments);
  var payload;
  try {
    payload = parse_rawbytes(args);
    handle_command(payload);
  } catch (err) {
    error_json(null, String(err));
  }
}

function parse_rawbytes(bytes) {
  if (!bytes || !bytes.length) {
    throw new Error("No raw bytes were received.");
  }
  var chars = [];
  var i;
  for (i = 0; i < bytes.length; i++) {
    if (typeof bytes[i] !== "number") {
      throw new Error("Raw byte list contained non-numeric value.");
    }
    if (bytes[i] === 0) {
      continue;
    }
    chars.push(String.fromCharCode(bytes[i]));
  }
  var raw = chars.join("");
  if (!raw || raw.charAt(0) !== "{") {
    throw new Error("Raw UDP payload is not a JSON object.");
  }
  return JSON.parse(raw);
}

function api(path) {
  return new LiveAPI(path);
}

function track_api(trackIndex) {
  return api("live_set tracks " + trackIndex);
}

function clip_slot_api(trackIndex, clipSlotIndex) {
  return api("live_set tracks " + trackIndex + " clip_slots " + clipSlotIndex);
}

function clip_api(trackIndex, clipSlotIndex) {
  var slot = clip_slot_api(trackIndex, clipSlotIndex);
  var hasClip = slot.get("has_clip");
  var clipPresent = Array.isArray(hasClip) ? hasClip[0] : hasClip;
  if (!clipPresent) {
    throw new Error("Clip slot " + clipSlotIndex + " on track " + trackIndex + " has no clip.");
  }
  return api("live_set tracks " + trackIndex + " clip_slots " + clipSlotIndex + " clip");
}

function mixer_parameter(trackIndex, parameterName) {
  var mixer = api("live_set tracks " + trackIndex + " mixer_device");
  return api(mixer.path + " " + parameterName);
}

function send_parameter(trackIndex, sendIndex) {
  var mixer = api("live_set tracks " + trackIndex + " mixer_device");
  return api(mixer.path + " sends " + sendIndex);
}

function device_parameter(trackIndex, deviceIndex, parameterIndex) {
  return api("live_set tracks " + trackIndex + " devices " + deviceIndex + " parameters " + parameterIndex);
}

function set_param_value(parameterApi, value) {
  parameterApi.set("value", value);
}

function note_insert(payload) {
  var clip = clip_api(payload.track_index, payload.clip_slot_index);
  var dict = new Dict();
  dict.parse(JSON.stringify(payload.notes));
  clip.call("add_new_notes", dict.name);
  return { inserted_notes: payload.notes.length };
}

function create_midi_clip(payload) {
  var slot = clip_slot_api(payload.track_index, payload.clip_slot_index);
  slot.call("create_clip", payload.length_beats);
  return { clip_length_beats: payload.length_beats };
}

function fire_clip(payload) {
  var slot = clip_slot_api(payload.track_index, payload.clip_slot_index);
  slot.call("fire");
  return { fired: true };
}

function stop_track(payload) {
  var track = track_api(payload.track_index);
  track.call("stop_all_clips");
  return { stopped: true };
}

function set_note_velocity(payload) {
  var clip = clip_api(payload.track_index, payload.clip_slot_index);
  var dict = new Dict();
  dict.parse(
    JSON.stringify([
      {
        pitch: payload.pitch,
        start_time: payload.start_time,
        duration: payload.duration,
        velocity: payload.velocity,
        mute: false
      }
    ])
  );
  clip.call("apply_note_modifications", dict.name);
  return { updated_velocity: payload.velocity };
}

function create_automation(payload) {
  var clip = clip_api(payload.track_index, payload.clip_slot_index);
  var parameter = device_parameter(payload.track_index, payload.device_index, payload.parameter_index);
  clip.call("clear_envelope", parameter.path);
  clip.call("create_automation_envelope", parameter.path);

  var envelope = api(parameter.path + " automation_envelope");
  for (var i = 0; i < payload.points.length; i++) {
    var point = payload.points[i];
    envelope.call("insert_step", point.time, point.value);
  }
  return { automation_points: payload.points.length };
}

function set_track_volume(payload) {
  var volume = mixer_parameter(payload.track_index, "volume");
  set_param_value(volume, payload.value);
  return { value: payload.value };
}

function set_track_mute(payload) {
  var track = track_api(payload.track_index);
  track.set("mute", payload.value ? 1 : 0);
  return { value: payload.value };
}

function set_track_solo(payload) {
  var track = track_api(payload.track_index);
  track.set("solo", payload.value ? 1 : 0);
  return { value: payload.value };
}

function set_track_pan(payload) {
  var pan = mixer_parameter(payload.track_index, "panning");
  set_param_value(pan, payload.value);
  return { value: payload.value };
}

function set_send_level(payload) {
  var send = send_parameter(payload.track_index, payload.send_index);
  set_param_value(send, payload.value);
  return { send_index: payload.send_index, value: payload.value };
}

function set_device_parameter(payload) {
  var parameter = device_parameter(payload.track_index, payload.device_index, payload.parameter_index);
  set_param_value(parameter, payload.value);
  return { value: payload.value };
}

function set_eq3(payload) {
  var baseTrack = payload.track_index;
  var deviceIndex = payload.device_index || 0;
  var mapping = {
    low_gain: 1,
    mid_gain: 2,
    high_gain: 3,
    low_on: 4,
    mid_on: 5,
    high_on: 6
  };
  var changed = 0;
  var field;

  for (field in mapping) {
    if (!mapping.hasOwnProperty(field)) {
      continue;
    }
    if (payload[field] === undefined) {
      continue;
    }
    var parameter = device_parameter(baseTrack, deviceIndex, mapping[field]);
    set_param_value(parameter, payload[field]);
    changed += 1;
  }
  return { changed_fields: changed };
}

function set_eq8_band_gain(payload) {
  var parameterIndex = payload.band;
  var parameter = device_parameter(payload.track_index, payload.device_index, parameterIndex);
  set_param_value(parameter, payload.gain);
  return { band: payload.band, gain: payload.gain };
}

function numeric_property_value(value, propertyName) {
  var parsed = Array.isArray(value) ? value[0] : value;
  var numberValue = Number(parsed);
  if (isNaN(numberValue)) {
    throw new Error("Could not parse numeric value for " + propertyName + ".");
  }
  return numberValue;
}

function set_tempo(payload) {
  var song = api("live_set");
  song.set("tempo", payload.bpm);
  return {
    requested_bpm: payload.bpm,
    current_bpm: numeric_property_value(song.get("tempo"), "tempo")
  };
}

function get_tempo(_payload) {
  var song = api("live_set");
  return { bpm: numeric_property_value(song.get("tempo"), "tempo") };
}

function set_global_key(payload) {
  var song = api("live_set");
  song.set("root_note", payload.root_note);
  if (payload.scale_name !== undefined) {
    song.set("scale_name", payload.scale_name);
  }
  if (payload.scale_intervals !== undefined) {
    song.set("scale_intervals", payload.scale_intervals.join(" "));
  }
  return {
    root_note: payload.root_note,
    scale_name: payload.scale_name || null,
    scale_intervals: payload.scale_intervals || null
  };
}

function get_track_count(_payload) {
  var song = api("live_set");
  return { track_count: song.getcount("tracks") };
}

var handlers = {
  note_insert: note_insert,
  create_midi_clip: create_midi_clip,
  fire_clip: fire_clip,
  stop_track: stop_track,
  set_note_velocity: set_note_velocity,
  create_automation: create_automation,
  set_track_volume: set_track_volume,
  set_track_mute: set_track_mute,
  set_track_solo: set_track_solo,
  set_track_pan: set_track_pan,
  set_send_level: set_send_level,
  set_device_parameter: set_device_parameter,
  set_eq3: set_eq3,
  set_eq8_band_gain: set_eq8_band_gain,
  set_tempo: set_tempo,
  set_global_key: set_global_key,
  get_track_count: get_track_count,
  get_tempo: get_tempo
};

function handle_command(message) {
  var id = message.id || null;
  var command = message.command;
  var payload = message.payload || {};

  if (!command || !handlers[command]) {
    throw new Error("Unsupported command: " + command);
  }
  var result = handlers[command](payload);
  success_json(id, result);
}
