\version "2.24.0"

\header {
  title = "4th Money Beat"
}

drumMusic = \drummode {
  \tempo 4 = 90
  \time 4/4
  <hh bd>4 <hh sn>4 <hh bd>4 <hh sn>4 | <hh bd>4 <hh sn>4 <hh bd>4 <hh sn>4 | <hh bd>4 <hh sn>4 <hh bd>4 <hh sn>4 | <hh bd>4 <hh sn>4 <hh bd>4 <hh sn>4 | <hh bd>4 r4
  \bar "|."
}

\score {
  \new DrumStaff <<
    \new DrumVoice { \voiceOne \drumMusic }
  >>
  \layout { }
  \midi { }
}
