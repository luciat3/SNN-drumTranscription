\version "2.24.0"

\header {
  title = "8th Money Beat"
}

drumMusic = \drummode {
  \tempo 4 = 90
  \time 4/4
  bd8 hh8 sn8 hh8 bd8 hh8 r8 sn8 | <hh bd>8 hh8 r8 sn8 hh8 <hh bd>8 hh8 <hh sn>8 | bd8 hh8 sn8 r8 bd8 hh8 sn8 r8 | <hh bd>8 sn8 r8 bd8 hh8 sn8 r8 <hh bd>8 | r8
  \bar "|."
}

\score {
  \new DrumStaff <<
    \new DrumVoice { \voiceOne \drumMusic }
  >>
  \layout { }
  \midi { }
}
