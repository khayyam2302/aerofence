#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "opendroneid.h"
#include "odid_wifi.h"

static void hex(const uint8_t *b, int n){ for(int i=0;i<n;i++) printf("%02x", b[i]); }

typedef struct { const char *id; double lat, lon; float dir, sh, sv, ab, ag, h, ts; double olat, olon; int hacc, vacc; } Case;

int main(void){
  Case cs[] = {
    {"1581F5FKD229400D7Q2F", 26.2187123, 50.0525456, 87.5f, 9.75f, -1.5f, 120.5f, 115.0f, 60.0f, 1234.5f, 26.2201, 50.0511, 10, 4},
    {"TESTUAS0000000000001", -33.8688197, 151.2092955, 270.0f, 70.25f, 2.0f, 30.0f, 32.5f, 15.5f, 3599.9f, -33.8690, 151.2100, 11, 5},
    {"ABC123", 45.0, -122.5, 359.0f, 0.0f, 0.0f, -1000.0f, 400.0f, -1000.0f, 0.0f, 0.0, 0.0, 9, 3},
    {"EDGE-SPEED", 51.4775, -0.4614, 181.0f, 120.0f, -10.0f, 900.0f, 950.0f, 200.0f, 1800.3f, 51.4700, -0.4600, 12, 6},
  };
  int n = sizeof(cs)/sizeof(cs[0]);
  printf("[\n");
  for(int k=0;k<n;k++){
    Case c = cs[k];
    ODID_UAS_Data u; odid_initUasData(&u);
    u.BasicID[0].IDType = ODID_IDTYPE_SERIAL_NUMBER; u.BasicID[0].UAType = ODID_UATYPE_HELICOPTER_OR_MULTIROTOR;
    strncpy(u.BasicID[0].UASID, c.id, sizeof(u.BasicID[0].UASID)); u.BasicIDValid[0]=1;
    ODID_Location_data *L=&u.Location; L->Status=ODID_STATUS_AIRBORNE; L->Direction=c.dir; L->SpeedHorizontal=c.sh;
    L->SpeedVertical=c.sv; L->Latitude=c.lat; L->Longitude=c.lon; L->AltitudeBaro=c.ab; L->AltitudeGeo=c.ag;
    L->HeightType=ODID_HEIGHT_REF_OVER_TAKEOFF; L->Height=c.h; L->HorizAccuracy=c.hacc; L->VertAccuracy=c.vacc;
    L->BaroAccuracy=ODID_VER_ACC_10_METER; L->SpeedAccuracy=ODID_SPEED_ACC_1_METERS_PER_SECOND; L->TSAccuracy=ODID_TIME_ACC_0_2_SECOND;
    L->TimeStamp=c.ts; u.LocationValid=1;
    u.System.OperatorLatitude=c.olat; u.System.OperatorLongitude=c.olon; u.System.AreaCount=1; u.System.AreaCeiling=-1000; u.System.AreaFloor=-1000;
    u.System.OperatorAltitudeGeo=-1000; u.System.Timestamp=0; u.SystemValid=1;
    ODID_Location_encoded le; encodeLocationMessage(&le,L);
    ODID_Location_data ld; decodeLocationMessage(&ld,&le);
    ODID_BasicID_encoded be; encodeBasicIDMessage(&be,&u.BasicID[0]);
    ODID_System_encoded se; encodeSystemMessage(&se,&u.System);
    ODID_System_data sd; decodeSystemMessage(&sd,&se);
    uint8_t pack[300]; int pl = odid_message_build_pack(&u, pack, sizeof(pack));
    uint8_t bf[600]; char mac[6]={0x02,0x11,0x22,0x33,0x44,(char)k};
    int bl = odid_wifi_build_message_pack_beacon_frame(&u, mac, "RID-TEST", 8, 100, (uint8_t)(10+k), bf, sizeof(bf));
    uint8_t nf[600]; int nl = odid_wifi_build_message_pack_nan_action_frame(&u, mac, (uint8_t)(20+k), nf, sizeof(nf));
    printf(" {\"case\": %d, \"uas_id\": \"%s\",\n  \"location_msg\": \"", k, c.id); hex((uint8_t*)&le,25);
    printf("\",\n  \"basic_id_msg\": \""); hex((uint8_t*)&be,25);
    printf("\",\n  \"system_msg\": \""); hex((uint8_t*)&se,25);
    printf("\",\n  \"pack\": \""); hex(pack,pl);
    printf("\",\n  \"beacon_frame\": \""); hex(bf,bl);
    printf("\", \"beacon_counter\": %d,\n  \"nan_frame\": \"", 10+k); hex(nf,nl);
    printf("\", \"nan_counter\": %d,\n", 20+k);
    printf("  \"expected\": {\"status\": %d, \"direction_deg\": %.6f, \"speed_h_ms\": %.6f, \"speed_v_ms\": %.6f, \"lat\": %.9f, \"lon\": %.9f, "
           "\"alt_baro_m\": %.6f, \"alt_geo_m\": %.6f, \"height_m\": %.6f, \"h_acc\": %d, \"v_acc\": %d, \"rid_ts_tenths\": %d, "
           "\"operator_lat\": %.9f, \"operator_lon\": %.9f, \"id_type\": %d, \"ua_type\": %d}}%s\n",
           ld.Status, ld.Direction, ld.SpeedHorizontal, ld.SpeedVertical, ld.Latitude, ld.Longitude, ld.AltitudeBaro, ld.AltitudeGeo,
           ld.Height, ld.HorizAccuracy, ld.VertAccuracy, (int)(ld.TimeStamp*10+0.5), sd.OperatorLatitude, sd.OperatorLongitude,
           u.BasicID[0].IDType, u.BasicID[0].UAType, k<n-1?",":"");
  }
  printf("]\n");
  return 0;
}
