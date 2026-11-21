// Type-00R Takemikazuchi over Ronin: do not draw the local player's own pilot while they board it.
//
// Why: the game shows boarding from an outside camera on the player's own pilot, who climbs into a hatch that
// opens.  On the Takemikazuchi nothing opens, and a pilot is a third of the machine's height, so the pilot
// sinks into the chest plate.
//
// What this does, and all it does: on this player's own screen, while the game's "boarding" flag is set on the
// local player and the titan being boarded is Ronin or Ronin Prime, the local player's model and the weapons it
// carries are not drawn; they are drawn again the moment the flag clears, the player becomes the titan, dies
// or leaves.  Client side only: nothing is sent to the server, nothing a server reads is changed, and no other
// player sees any difference.

global function TSFEmbark_Init

struct
{
	bool hiding = false
	bool refusalPrinted = false
	bool acceptedPrinted = false
} file

void function TSFEmbark_Init()
{
	if ( IsLobby() )
		return

	AddCinematicEventFlagChangedCallback( CE_FLAG_EMBARK, TSFEmbark_FlagChanged )
}

void function TSFEmbark_FlagChanged( entity player )
{
	if ( file.hiding )
		return

	if ( !IsValid( player ) )
		return

	if ( player != GetLocalClientPlayer() )
		return

	if ( IsWatchingReplay() )
		return

	if ( ( player.GetCinematicEventFlags() & CE_FLAG_EMBARK ) == 0 )
		return

	if ( player.IsTitan() )
		return

	entity titan = player.GetPetTitan()
	if ( !IsValid( titan ) )
		return

	asset titanModel = titan.GetModelName()
	if ( titanModel != $"models/titans/light/titan_light_locust.mdl" && titanModel != $"models/titans/light/titan_light_ronin_prime.mdl" )
		return

	thread TSFEmbark_HidePilot( player )
}

void function TSFEmbark_HidePilot( entity player )
{
	array<entity> hidden = [ player ]
	foreach ( entity weapon in player.GetMainWeapons() )
		hidden.append( weapon )

	OnThreadEnd(
		function() : ( hidden )
		{
			foreach ( entity ent in hidden )
				TSFEmbark_SetDrawn( ent, true )

			file.hiding = false
			printt( "TSFEmbark: boarding over, pilot drawn again" )
		}
	)

	file.hiding = true
	printt( "TSFEmbark: boarding Ronin, not drawing the local pilot" )

	player.EndSignal( "OnDestroy" )

	while ( IsAlive( player ) && !player.IsTitan() && ( player.GetCinematicEventFlags() & CE_FLAG_EMBARK ) != 0 )
	{
		// every frame, in case an update from the server puts the model back
		foreach ( entity ent in hidden )
			TSFEmbark_SetDrawn( ent, false )

		WaitFrame()
	}
}

void function TSFEmbark_SetDrawn( entity ent, bool drawn )
{
	if ( !IsValid( ent ) )
		return

	// Hide() / Show() are refused on an entity that comes from the server ("You can only call
	// SetInvisibleForLocalPlayer() on client-side entities", seen in the log on 2026-11-15), so the entity's
	// own draw switch is used.  If the game refuses that too, the refusal is printed once and nothing changes.
	try
	{
		if ( drawn )
			ent.EnableDraw()
		else
			ent.DisableDraw()

		if ( !file.acceptedPrinted )
		{
			file.acceptedPrinted = true
			printt( "TSFEmbark: the game accepted DisableDraw / EnableDraw" )
		}
	}
	catch ( ex )
	{
		if ( !file.refusalPrinted )
		{
			file.refusalPrinted = true
			printt( "TSFEmbark: the game refused DisableDraw / EnableDraw on this entity:", ex )
		}
	}
}
