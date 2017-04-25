set i 0
for file in *.png
  mv -v -- "$file" "croco"$i.png
  set i (math $i + 1)
end
