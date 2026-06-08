/**
 * RB Sim Bridge
 *
 * Long-running Node.js process that exposes the Pokemon Showdown simulator
 * (@pkmn/sim) via a JSON-line stdin/stdout protocol. Designed to be spawned
 * once by the Python AI and reused across many turn simulations.
 *
 * Protocol
 * --------
 * Each line of stdin is a JSON object with a "cmd" field. Each response is a
 * single JSON line on stdout. Errors are reported as {ok: false, error: ...}.
 *
 * Commands
 * --------
 *   {cmd: "ping"}
 *       -> {ok: true, pong: true}
 *
 *   {cmd: "init", format: "gen9customgame",
 *    p1: {name, team}, p2: {name, team}, seed?: [a,b,c,d]}
 *       -> {ok: true, state: <serialized>, requests: {p1, p2},
 *           ended: bool, winner: string|null, turn: int}
 *
 *   {cmd: "step", state: <serialized>,
 *    p1_choice: "move 1" | "switch 2" | "default" | null,
 *    p2_choice: "...", seed?: [a,b,c,d]}
 *       -> {ok: true, state, requests, ended, winner, turn}
 *
 *   {cmd: "requests", state: <serialized>}
 *       -> {ok: true, requests: {p1, p2}, ended, winner, turn}
 *
 *   {cmd: "quit"}
 *       -> process exits 0
 *
 * Notes
 * -----
 *   - Teams are passed as Showdown "packed" strings (see Teams.pack).
 *   - "format" should generally be "gen9customgame" so any team is accepted
 *     (Random Battle level / EVs / IVs are encoded in the packed team).
 *   - p1_choice / p2_choice follow Showdown's choice protocol: "move 1",
 *     "move 1 terastallize", "switch 3", "default", etc.
 *   - If a side has no active request (e.g. it already fainted and there's
 *     only one side to switch), pass null for that choice.
 *
 * @license MIT
 */

'use strict';

const readline = require('readline');
const {Battle, State, Teams} = require('@pkmn/sim');

const STDIN = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
});

function writeResponse(obj) {
    process.stdout.write(JSON.stringify(obj) + '\n');
}

function writeError(error, extra) {
    const payload = {ok: false, error: String(error && error.message || error)};
    if (extra) Object.assign(payload, extra);
    writeResponse(payload);
}

/**
 * Parse a packed team string. Accept "packed" (string) or raw set array.
 */
function parseTeam(team) {
    if (Array.isArray(team)) return team;
    if (typeof team === 'string') {
        const parsed = Teams.unpack(team);
        if (!parsed) throw new Error('Failed to unpack team: ' + team.slice(0, 80));
        return parsed;
    }
    throw new Error('Unsupported team type: ' + typeof team);
}

/**
 * Strip transient fields from requests so the Python side gets a compact view.
 */
function compactRequest(req) {
    if (!req) return null;
    if (req.wait) return {wait: true};
    if (req.forceSwitch) {
        return {
            forceSwitch: req.forceSwitch,
            side: req.side ? compactSide(req.side) : null,
        };
    }
    if (req.teamPreview) {
        return {teamPreview: true, side: req.side ? compactSide(req.side) : null};
    }
    if (req.active) {
        return {
            active: req.active,
            side: req.side ? compactSide(req.side) : null,
        };
    }
    return req;
}

function compactSide(side) {
    if (!side) return null;
    return {
        name: side.name,
        id: side.id,
        pokemon: (side.pokemon || []).map((p) => ({
            ident: p.ident,
            details: p.details,
            condition: p.condition,
            active: !!p.active,
            stats: p.stats,
            moves: p.moves,
            ability: p.ability,
            baseAbility: p.baseAbility,
            item: p.item,
            pokeball: p.pokeball,
            teraType: p.teraType,
            terastallized: p.terastallized,
        })),
    };
}

function snapshotBattle(battle, options) {
    const requests = battle.getRequests(battle.requestState);
    const includeLog = options && options.includeLog;
    const snap = {
        state: State.serializeBattle(battle),
        requests: {
            p1: compactRequest(requests[0]),
            p2: compactRequest(requests[1]),
        },
        ended: battle.ended,
        winner: battle.winner || null,
        turn: battle.turn,
        request_state: battle.requestState,
    };
    if (includeLog) {
        snap.log_tail = (battle.log || []).slice(-40);
    }
    return snap;
}

function buildBattle({format, p1, p2, seed}) {
    const formatid = format || 'gen9customgame';
    const options = {
        formatid,
        seed: seed || undefined,
        strictChoices: false,
        debug: false,
        deserialized: false,
    };
    options.p1 = {
        name: (p1 && p1.name) || 'p1',
        team: parseTeam(p1.team),
    };
    options.p2 = {
        name: (p2 && p2.name) || 'p2',
        team: parseTeam(p2.team),
    };
    const battle = new Battle(options);
    // Force-start so we get an immediate request to act on.
    if (!battle.started) battle.start();
    return battle;
}

function reviveBattle(state, seed) {
    const battle = State.deserializeBattle(state);
    if (seed) battle.prng = new (require('@pkmn/sim').PRNG)(seed);
    return battle;
}

function applyChoices(battle, p1Choice, p2Choice) {
    // For each side, if a request is active, send a choice. Defaults are
    // tolerated to let the engine pick the first legal action.
    if (p1Choice !== null && p1Choice !== undefined) {
        battle.choose('p1', String(p1Choice));
    }
    if (p2Choice !== null && p2Choice !== undefined) {
        battle.choose('p2', String(p2Choice));
    }
}

function handle(cmd) {
    switch (cmd.cmd) {
        case 'ping':
            return {ok: true, pong: true};

        case 'init': {
            const battle = buildBattle(cmd);
            return {ok: true, ...snapshotBattle(battle, {includeLog: cmd.include_log})};
        }

        case 'step': {
            const battle = reviveBattle(cmd.state, cmd.seed);
            applyChoices(battle, cmd.p1_choice, cmd.p2_choice);
            return {ok: true, ...snapshotBattle(battle, {includeLog: cmd.include_log})};
        }

        case 'requests': {
            const battle = reviveBattle(cmd.state);
            return {ok: true, ...snapshotBattle(battle, {includeLog: cmd.include_log})};
        }

        case 'quit':
            process.exit(0);
            return null;

        default:
            throw new Error('Unknown cmd: ' + cmd.cmd);
    }
}

STDIN.on('line', (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let parsed;
    try {
        parsed = JSON.parse(trimmed);
    } catch (err) {
        writeError('Invalid JSON: ' + err.message);
        return;
    }
    try {
        const reply = handle(parsed);
        if (reply) writeResponse(reply);
    } catch (err) {
        writeError(err, {cmd: parsed && parsed.cmd});
    }
});

STDIN.on('close', () => process.exit(0));
