adb.exe -s emulator-5554 shell am start -n com.customscopecommunity.crosshairpro/.MainActivity
adb.exe -s emulator-5554 shell sleep 2
adb.exe -s emulator-5554 shell input tap 543 1132
adb.exe -s emulator-5554 shell am start -n com.tencent.tmgp.cod/com.tencent.tmgp.cod.CODMainActivity