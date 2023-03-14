adb.exe -s emulator-5556 push .\run.sh /data/local/tmp
adb.exe -s emulator-5556 shell chmod a+x /data/local/tmp/run.sh
adb.exe -s emulator-5556 shell sync
adb.exe -s emulator-5556 shell sh /data/local/tmp/run.sh