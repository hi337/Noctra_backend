from muselsl import stream, list_muses

muses = list_muses()
stream(muses[0]['address'], ppg_enabled=True, acc_enabled=True, gyro_enabled=True)