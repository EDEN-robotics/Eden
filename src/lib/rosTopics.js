// Browser-local ROS-style pub/sub bus.
// Mirrors rclcpp topic semantics (latched last message, rate counter,
// subscribers) without needing a real ROS graph. Makes the simulator
// *feel* like it's talking to a ROS 2 stack — same names, same message
// shapes — without leaving the browser.

const topics = new Map()

function touch(name) {
  if (!topics.has(name)) {
    topics.set(name, {
      name,
      lastMsg: null,
      msgCount: 0,
      rate: 0,           // EMA Hz
      lastPublishMs: 0,
      subs: new Set(),
    })
  }
  return topics.get(name)
}

export function publish(name, msg) {
  const t = touch(name)
  const now = performance.now()
  if (t.lastPublishMs > 0) {
    const dt = (now - t.lastPublishMs) / 1000
    if (dt > 0) t.rate = 0.85 * t.rate + 0.15 * (1 / dt)
  }
  t.lastPublishMs = now
  t.msgCount += 1
  t.lastMsg = msg
  for (const sub of t.subs) {
    try { sub(msg) } catch (err) { console.warn('[rosTopics] subscriber threw:', err) }
  }
}

export function subscribe(name, cb) {
  const t = touch(name)
  t.subs.add(cb)
  return () => t.subs.delete(cb)
}

export function listTopics() {
  return Array.from(topics.values()).map((t) => ({
    name: t.name,
    rate: t.rate,
    msgCount: t.msgCount,
    lastPublishMs: t.lastPublishMs,
    lastMsg: t.lastMsg,
    subscribers: t.subs.size,
  }))
}

// Message type constructors (match ROS 2 geometry_msgs / sensor_msgs / nav_msgs)
export function Twist(lin = 0, ang = 0) {
  return {
    _type: 'geometry_msgs/Twist',
    linear: { x: lin, y: 0, z: 0 },
    angular: { x: 0, y: 0, z: ang },
  }
}

export function Odometry({ x, y, heading, linVel, angVel, ts }) {
  return {
    _type: 'nav_msgs/Odometry',
    header: { stamp: ts, frame_id: 'odom' },
    child_frame_id: 'base_link',
    pose: {
      position: { x, y, z: 0 },
      orientation: yawToQuat(heading),
    },
    twist: {
      linear: { x: linVel, y: 0, z: 0 },
      angular: { x: 0, y: 0, z: angVel },
    },
  }
}

export function LaserScan({ angle_min, angle_max, ranges, range_max, ts }) {
  return {
    _type: 'sensor_msgs/LaserScan',
    header: { stamp: ts, frame_id: 'base_scan' },
    angle_min,
    angle_max,
    angle_increment: (angle_max - angle_min) / Math.max(1, ranges.length - 1),
    range_min: 0.05,
    range_max,
    ranges,
  }
}

export function TFMessage({ x, y, heading, ts }) {
  return {
    _type: 'tf2_msgs/TFMessage',
    transforms: [{
      header: { stamp: ts, frame_id: 'map' },
      child_frame_id: 'base_link',
      transform: {
        translation: { x, y, z: 0 },
        rotation: yawToQuat(heading),
      },
    }],
  }
}

function yawToQuat(yaw) {
  const half = yaw / 2
  return { x: 0, y: 0, z: Math.sin(half), w: Math.cos(half) }
}
