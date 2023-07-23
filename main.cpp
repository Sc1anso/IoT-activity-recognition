#include <fstream>
#include <string>
#include <sstream>
#include <Arduino.h>
#include <M5Stack.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <map>
#include <ArduinoEigen.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiMulti.h>

using namespace std;

int list_count = 0;
int n_records[6] = {500, 25, 25, 100, 100, 500}; //n max records to save
int records = 0;
const int N_WEIGHTS = 100;
const int N_FEATURES = 118;

bool execute = false;
bool clear_screen = false;
bool local_exec = false;

float imu_data[6];
float w_mtx[N_WEIGHTS][N_FEATURES];
int lbls[N_WEIGHTS];



string path1 = "/new_body_gyro_x.txt";
string path2 = "/new_body_gyro_y.txt";
string path3 = "/new_body_gyro_z.txt";
string path4 = "/new_total_acc_x.txt";
string path5 = "/new_total_acc_y.txt";
string path6 = "/new_total_acc_z.txt";
string path7 = "/new_y.txt";

string gyroX = "";
string gyroY = "";
string gyroZ = "";
string accX = "";
string accY = "";
string accZ = "";
string y = "";

vector<float> gyroXv;
vector<float> gyroYv;
vector<float> gyroZv;
vector<float> accXv;
vector<float> accYv;
vector<float> accZv;
vector<float> timev;

WiFiMulti wifiMulti;

string menu_items[9] = {"walking", "walking upstairs", "walking downstairs", "sitting", "standing", "laying", "testing KMEANS", "testing SOM local", "testing SOM online preproc"};

int act_count = 1; // activity integer value

string vector_to_string(vector<float> vec) {
  //ostringstream stream;
  string tmp = to_string(vec[0]);
  //stream << vec[0];
  for (int i = 1; i < vec.size(); i++) {
    tmp += "_" + to_string(vec[i]);
    //stream << tmp;
  }
  return tmp;
}

vector<float> medianFilter(vector<float> input, int windowSize){
    vector<float> output(input.size());
    // Apply median filter to input signal
    for (int i = 0; i < input.size(); i++) {
        // Create a temporary buffer to hold the surrounding samples
        vector<float> buffer;

        // Fill the buffer with the surrounding samples
        for (int j = i - windowSize / 2; j <= i + windowSize / 2; j++) {
            if (j >= 0 && j < input.size()) {
                buffer.push_back(input[j]);
            }
        }

        // Sort the buffer
        sort(buffer.begin(), buffer.end());

        // Replace the current sample with the median value
        output[i] = buffer[buffer.size() / 2];
    }
    return output;
}

vector<float> thirdordButtFilt(vector<float> signal, float dt, float cutoffFrequency) {
    float b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0, b4 = 0.0;
    float a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0;
    float alpha = tan(M_PI * cutoffFrequency * dt);

    b0 = 1.0 / (1.0 + sqrt(2.0) * alpha + alpha * alpha);
    b1 = 2.0 * b0;
    b2 = b0;
    a1 = 2.0 * b0 * (alpha * alpha - 1.0);
    a2 = b0 * (1.0 - sqrt(2.0) * alpha + alpha * alpha);

    vector<float> output(signal.size());
    float x1 = 0.0, x2 = 0.0, x3 = 0.0, x4 = 0.0;
    float y1 = 0.0, y2 = 0.0, y3 = 0.0, y4 = 0.0;
    for (int i = 0; i < signal.size(); i++) {
        x1 = x2;
        x2 = x3;
        x3 = x4;
        x4 = signal[i];
        y4 = b0 * x4 + b1 * x3 + b2 * x2 + b3 * x1 - a1 * y1 - a2 * y2 - a3 * y3 - a4 * y4;
        y3 = y2;
        y2 = y1;
        y1 = y4;
        output[i] = y4;
    }

    return output;
}

// Function to calculate the Butterworth filter coefficients
vector<float> butterworthCoeffs(float cutoff, int order) {
    vector<float> coeffs;
    for (int i = 0; i <= order; i++) {
        coeffs.push_back(1.0 / (1.0 + pow(cutoff, 2 * i)));
    }
    return coeffs;
}

// Function to apply the Butterworth filter
vector<float> butterworthFilter(vector<float> data, float cutoff, int order) {
    vector<float> filteredData(data.size(), 0);
    vector<float> coeffs = butterworthCoeffs(cutoff, order);
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j <= order; j++) {
            if (i - j >= 0) {
                filteredData[i] += coeffs[j] * data[i - j];
            }
        }
    }
    return filteredData;
}

vector<float> getGravity(vector<float> total, vector<float> body){
    vector<float> gravity;
    for (size_t i = 0; i < total.size(); i++)
    {
        gravity.push_back(total[i] - body[i]);
    }
    return gravity;
}

// WRITE FILE TO SD
void appendFile(fs::FS &fs, const char * path, const char * message){
    Serial.printf("Appending to file: %s\n", path);

    File file = fs.open(path, FILE_APPEND);
    if(!file){
        Serial.println("Failed to open file for appending");
        return;
    }
    if(file.print(message)){
        Serial.println("Message appended");
    } else {
        Serial.println("Append failed");
    }
    file.close();
}

// CALCULATE VELOCITY
vector<float> calculate_velocity( vector<float> acceleration,  vector<float> time) {
    vector<float> velocity;
    for (int i = 0; i < acceleration.size() - 1; i++) {
        float vel = (acceleration[i + 1] - acceleration[i]) / (time[i + 1] - time[i]);
        velocity.push_back(vel);
    }
    return velocity;
}

// CALCULATE JERK
vector<float> calculate_jerk( vector<float> velocity,  vector<float> time) {
    vector<float> jerk;
    for (int i = 0; i < velocity.size() - 1; i++) {
        float jer = (velocity[i + 1] - velocity[i]) / (time[i + 1] - time[i]);
        jerk.push_back(jer);
    }
    return jerk;
}

// CALCULATE MEAN
float average(vector<float>  v){
    if(v.empty()){
        return 0;
    }

    auto  count = static_cast<float>(v.size());
    return accumulate(v.begin(), v.end(), 0.0) / count;
}

// CALCULATE STD DEV
float std_dev(vector<float>  v, float  mean){
    float sq_sum = inner_product(v.begin(), v.end(), v.begin(), 0.0);
    float stdev = sqrt(sq_sum / v.size() - mean * mean);
    return stdev;
}

// CALCULATE MEDIAN ABSOLUTE DEVIATION
float mad(vector<float> data) {
    // Find the median of the dataset
    int n = data.size();
    nth_element(data.begin(), data.begin() + n / 2, data.end());
    float median = data[n / 2];

    // Calculate the absolute deviation from the median
    vector<float> deviation;
    for (int i = 0; i < n; i++) {
        deviation.push_back(fabs(data[i] - median));
    }

    // Find the median of the absolute deviation
    nth_element(deviation.begin(), deviation.begin() + n / 2, deviation.end());
    float mad = deviation[n / 2];
    return mad;
}

float max_v(vector<float> v){
    return *max_element(v.begin(), v.end());
}

float min_v(vector<float> v){
    return *min_element(v.begin(), v.end());
}

// SIGNAL MAGNITUDE AREA
float signalMagnitudeArea(vector<float> x, vector<float> y, vector<float> z, int start, int end) {
    float sum = 0;
    int n = end - start;
    for (int i = start; i < end; i++) {
        float magnitude = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        sum += magnitude;
    }
    return sum / n;
}

// SIGNAL ENERGY
float signalEnergy(vector<float> signal, int start, int end) {
    float sum = 0;
    for (int i = start; i < end; i++) {
        sum += signal[i] * signal[i];
    }
    return sum / end;
}

// SIGNAL INTERQUARTILE RANGE
float signalInterquartileRange(vector<float> signal_o) {
    vector<float> signal(signal_o.begin(), signal_o.end());
    sort(signal.begin(), signal.end());
    /* M5.Lcd.setCursor(0, 162);
    M5.Lcd.printf("signal : %f - %f", signal_o[5], signal_o[96]);
    delay(2000); */

    // Find the 25th percentile
    int n = signal.size();
    float Q1 = signal[n / 4];

    // Find the 75th percentile
    float Q3 = signal[(3 * n) / 4];
    /* M5.Lcd.setCursor(0, 162);
    M5.Lcd.printf("n/4 : %d - 3*n/4: %d", n/4, (3*n)/4);
    M5.Lcd.setCursor(0, 182);
    M5.Lcd.printf("q1: %f - q3: %f", Q1, Q3);
    M5.Lcd.setCursor(0, 202); */

    // Calculate the IQR
    return Q3 - Q1;
    
}

// SIGNAL ENTROPY
float signalEntropy(vector<float> signal, int start, int end) {
    std::map<float, int> signal_histogram;
    for (int i = start; i < end; i++) {
        signal_histogram[signal[i]]++;
    }
    float entropy = 0;
    int n = end - start;
    for ( auto p : signal_histogram) {
        float p_i = (float) p.second / n;
        entropy += -p_i * log2(p_i);
    }
    return entropy;
}

// SIGNAL AR COEFFICIENT with order = 4
vector<float> signalAutoregressionCoefficientsBurg(vector<float> signal, int order) {
    int n = signal.size();
    vector<float> ar_coefficients(order, 0);
    vector<float> e(n, 0);
    vector<float> k(order, 0);
    vector<float> a(order + 1, 0);

    // Initialize e[0] and a[0]
    for (int i = 0; i < n; i++) {
        e[0] += signal[i] * signal[i];
    }
    a[0] = 1;

    // Iterate over order
    for (int i = 1; i <= order; i++) {
        // Compute k[i-1]
        float num = 0;
        float den = 0;
        for (int j = 0; j < n - i; j++) {
            num += signal[j] * signal[j + i];
            den += signal[j] * signal[j];
        }
        k[i - 1] = -2 * num / den;

        // Update a and e
        a[i] = k[i - 1];
        for (int j = 1; j <= i - 1; j++) {
            a[j] = a[j] + k[i - 1] * a[i - j];
        }
        e[i] = (1 - k[i - 1] * k[i - 1]) * e[i - 1];
    }

    // Compute AR coefficients
    for (int i = 1; i <= order; i++) {
        ar_coefficients[i - 1] = a[i];
    }
    return ar_coefficients;
}

// CORRELATION BETWEEN 2 SIGNALS
float correlationCoefficient(vector<float> signal1, vector<float> signal2) {
    int n = signal1.size();
    float mean1 = 0, mean2 = 0, stddev1 = 0, stddev2 = 0;
    float correlation = 0;
    // Compute mean and standard deviation of both signals
    for (int i = 0; i < n; i++) {
        mean1 += signal1[i];
        mean2 += signal2[i];
    }
    mean1 /= n;
    mean2 /= n;
    for (int i = 0; i < n; i++) {
        stddev1 += (signal1[i] - mean1) * (signal1[i] - mean1);
        stddev2 += (signal2[i] - mean2) * (signal2[i] - mean2);
        correlation += (signal1[i] - mean1) * (signal2[i] - mean2);
    }
    stddev1 = sqrt(stddev1 / n);
    stddev2 = sqrt(stddev2 / n);
    // Compute correlation coefficient
    correlation /= n * stddev1 * stddev2;
    return correlation;
}

// MAG CALCULATION
float signalMagnitude(float x, float y, float z) {
    // Calculate the signal magnitude
    return sqrt(x * x + y * y + z * z);
}

vector<float> getMagVector(vector<float> x_v, vector<float> y_v, vector<float> z_v){
    vector<float> output;
    for (size_t i = 0; i < x_v.size(); i++)
    {
        output.push_back(signalMagnitude(x_v[i], y_v[i], z_v[i]));
    }
    return output;
}

float getSMAfromMag(vector<float> mag){
    float output = 0;
    int n = mag.size();
    for (size_t i = 0; i < n; i++)
    {
        output += mag[i];
    }
    return output / n;
}

int winner_index(vector<float>& sample) {
    int winner = 0;
    float min_dist = 3.402823466e+38;
    for (int i = 0; i < N_WEIGHTS; i++) {
        float dist = 0;
        for (int j = 0; j < N_FEATURES; j++) {
            /* M5.Lcd.setCursor(0, 0);
            M5.Lcd.printf("sample: %f, j: %d", sample[j], j);
            M5.Lcd.setCursor(0, 20);
            M5.Lcd.printf("w_mtx: %f, j: %d", w_mtx[i][j], j);
            delay(1000); */
            dist += fabs(sample[j] - w_mtx[i][j]);
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            winner = i;
        }
    }
    return winner;
}

vector<float> split(const String& s, char delimiter) {
    vector<float> tokens;
    string token;
    stringstream stream(s.c_str());
    while (getline(stream, token, delimiter)) {
        tokens.push_back(stof(token));
    }
    return tokens;
}


// MAIN PROGRAM
void setup() {
    M5.begin();        // Init M5Core.  初始化 M5Core
    M5.Power.begin();  // Init Power module.  初始化电源
    
    wifiMulti.addAP(
        "wifi_name",
        "wifi_pass");

    M5.IMU.Init();  // Init IMU sensor.  初始化惯性传感器

    M5.Lcd.fillScreen(BLACK);  // Set the screen background color to black.
                               // 设置屏幕背景色为黑色
    M5.Lcd.setTextColor(
        GREEN, BLACK);  // Sets the foreground color and background color of the
                        // displayed text.  设置显示文本的前景颜色和背景颜色
    M5.Lcd.setTextSize(2);  // Set the font size.  设置字体大小

    if (N_WEIGHTS == 100)
    {
        File file = SD.open("/weights_lst_avg_bal_10.txt");

        int row = 0;
        int col = 0;
        String line;
        while(file.available()){
            char c = file.read();
            if(c == '\n'){
                //rows = row + 1;
                //cols = max(cols, col);
                col = 0;
                row++;
                line = "";
            }
            else{
                line += c;
                if(c == ' '){
                    w_mtx[row][col] = line.toFloat();
                    col++;
                    line = "";
                }
            }
        }
        file.close();

        file = SD.open("/map_lst_bal_10.txt");

        row = 0;
        //int col = 0;
        line = "";
        while(file.available()){
            char c = file.read();
            if(c == '\n'){
                //rows = row + 1;
                //cols = max(cols, col);
                //col = 0;
                row++;
                line = "";
                //delay(1000);
            }
            else{
                line += c;
                //M5.Lcd.setCursor(0, 0);
                //M5.Lcd.printf("val: %d", line.toInt());
                lbls[row] = line.toInt();
                // line = "";
            }
        }
        file.close();
    }
    else {
        File file;
        int row = 0;
        int col = 0;
        String line;
        for (int i = 1; i <= 10; i++)
        {
            string path = "/weights_lst_avg_bal_17_pt" + to_string(i) + ".txt";
            file = SD.open(path.c_str());

            while(file.available()){
                char c = file.read();
                if(c == '\n'){
                    //rows = row + 1;
                    //cols = max(cols, col);
                    col = 0;
                    row++;
                    line = "";
                }
                else{
                    line += c;
                    if(c == ' '){
                        w_mtx[row][col] = line.toFloat();
                        col++;
                        line = "";
                    }
                }
            }
            file.close();

            
        }
        file = SD.open("/map_lst_bal_17.txt");

        row = 0;
        //int col = 0;
        line = "";
        while(file.available()){
            char c = file.read();
            if(c == '\n'){
                //rows = row + 1;
                //cols = max(cols, col);
                //col = 0;
                row++;
                line = "";
                //delay(1000);
            }
            else{
                line += c;
                //M5.Lcd.setCursor(0, 0);
                //M5.Lcd.printf("val: %d", line.toInt());
                lbls[row] = line.toInt();
                // line = "";
            }
        }
        file.close();
        
    }
    
    

    int count = 0;

    while (wifiMulti.run() != WL_CONNECTED) {
        M5.Lcd.clear();
        M5.Lcd.setCursor(0, 0);
        if (count == 0)
        {
            M5.Lcd.printf("Connecting to wifi");
            count++;
        }
        else if (count == 1)
        {
            M5.Lcd.printf("Connecting to wifi.");
            count++;
        }
        else if (count == 2)
        {
            M5.Lcd.printf("Connecting to wifi..");
            count++;
        }
        else if (count == 3)
        {
            M5.Lcd.printf("Connecting to wifi...");
            count = 0;
        }
        delay(10);
    }
    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 0);
    M5.Lcd.printf("Connected to wifi");
    delay(1000);
    
}


void loop(){
    if (!execute)
    {
        clear_screen = false;
        M5.update();  // Read the press state of the key.  读取按键 A, B, C 的状态
        if (act_count < 7)
        {
            // set title
            M5.Lcd.setCursor(0, 0);
            M5.Lcd.printf("Add record to dataset for");
            M5.Lcd.setCursor(0, 25);
            M5.Lcd.printf("activity:");
            // set selcted activity
            M5.Lcd.setCursor(0, 50);
            M5.Lcd.printf("- %s", menu_items[act_count - 1].c_str());
            M5.Lcd.setCursor(0, 75);
            M5.Lcd.printf("number of records to record %d", n_records[act_count - 1]);
        } else {
            //M5.Lcd.clear();
            M5.Lcd.setCursor(0, 0);
            M5.Lcd.printf("Start");
            //M5.Lcd.setCursor(0, 25);
            //M5.Lcd.printf("activity:");
            // set selcted activity
            M5.Lcd.setCursor(0, 25);
            M5.Lcd.printf("- %s", menu_items[act_count - 1].c_str());
            /* M5.Lcd.setCursor(0, 75);
            M5.Lcd.printf("number of records to record %d", n_records[act_count - 1]); */
        }
        // set btn A label
        M5.Lcd.setCursor(41, 220);
        M5.Lcd.printf("Prev");
        // set btn B label
        M5.Lcd.setCursor(129, 220);
        M5.Lcd.printf("Start");
        // set btn C label
        M5.Lcd.setCursor(229, 220);
        M5.Lcd.printf("Next");

        // set btn Prev
        if (M5.BtnA.wasPressed())
        {
            if (act_count == 1)
            {
                act_count = 9;
            } 
            else
            {
                act_count--;
            }
            
            clear_screen = true;
        }

        // set btn Start
        if (M5.BtnB.wasPressed())
        {
            execute = true;
            clear_screen = true;
        }

        // set btn Next
        if (M5.BtnC.wasPressed())
        {
            if (act_count == 9)
            {
                act_count = 1;
            } 
            else
            {
                act_count++;
            }
            
            clear_screen = true;
        }
        y = to_string(act_count);
    } else if (execute) // branch executed if start button was pressed
    {
        clear_screen = false;
        if (act_count < 7)
        {
            if (records < n_records[act_count - 1])
            {
                // get sensor data
                M5.IMU.getGyroData(&imu_data[0], &imu_data[1], &imu_data[2]);
                M5.IMU.getAccelData(&imu_data[3], &imu_data[4], &imu_data[5]);

                M5.Lcd.setCursor(0, 142);
                M5.Lcd.printf("steps: %d", list_count);
                M5.Lcd.setCursor(0, 162);
                M5.Lcd.printf("Activity: %s", menu_items[act_count - 1].c_str());
                M5.Lcd.setCursor(0, 182);
                M5.Lcd.printf("Label: %d", act_count);
                
                list_count +=1;

                if (list_count <= 129)
                {
                    if (list_count == 129){
                        
                        appendFile(SD, path1.c_str(), (gyroX + "\n").c_str());
                        appendFile(SD, path2.c_str(), (gyroY + "\n").c_str());
                        appendFile(SD, path3.c_str(), (gyroZ + "\n").c_str());
                        appendFile(SD, path4.c_str(), (accX + "\n").c_str());
                        appendFile(SD, path5.c_str(), (accY + "\n").c_str());
                        appendFile(SD, path6.c_str(), (accZ + "\n").c_str());
                        appendFile(SD, path7.c_str(), (y + "\n").c_str());
                        list_count = 0;
                        records += 1;
                        gyroX = "";
                        gyroY = "";
                        gyroZ = "";
                        accX = "";
                        accY = "";
                        accZ = "";
                    }
                    else
                    {
                        gyroX += to_string(imu_data[0]) + "  ";
                        gyroY += to_string(imu_data[1]) + "  ";
                        gyroZ += to_string(imu_data[2]) + "  ";
                        accX += to_string(imu_data[3]) + "  ";
                        accY += to_string(imu_data[4]) + "  ";
                        accZ += to_string(imu_data[5]) + "  ";
                    }
                    M5.Lcd.setCursor(0, 202);
                    M5.Lcd.printf("Recorded records: %i/%i", records, n_records[act_count - 1]);
                }
            } else
            {
                execute = false;
                records = 0;
                list_count = 0;
                clear_screen = true;
                delay(2000);
            }
        } else
        {
            // get sensor data
            M5.IMU.getGyroData(&imu_data[0], &imu_data[1], &imu_data[2]);
            M5.IMU.getAccelData(&imu_data[3], &imu_data[4], &imu_data[5]);

            M5.Lcd.setCursor(0, 142);
            M5.Lcd.printf("steps: %d ", list_count);
            M5.Lcd.setCursor(0, 162);
            M5.Lcd.printf("Activity: %s", menu_items[act_count - 1].c_str());
            M5.Lcd.setCursor(0, 182);
            M5.Lcd.printf("act_count: %d", act_count);
            list_count +=1;

            if (list_count <= 129)
            {
                if (list_count == 129){
                    list_count = 0;
                    vector<float> features;

                    // KMEANS
                    if (act_count == 7)
                    {
                        gyroXv = medianFilter(gyroXv, gyroXv.size() / 2);
                        gyroYv = medianFilter(gyroYv, gyroYv.size() / 2);
                        gyroZv = medianFilter(gyroZv, gyroZv.size() / 2);
                        accXv = medianFilter(accXv, accXv.size() / 2);
                        accYv = medianFilter(accYv, accYv.size() / 2);
                        accZv = medianFilter(accZv, accZv.size() / 2);
                        
                        gyroXv = thirdordButtFilt(gyroXv, 1/41.29, 0.02);
                        gyroYv = thirdordButtFilt(gyroYv, 1/41.29, 0.02);
                        gyroZv = thirdordButtFilt(gyroZv, 1/41.29, 0.02);
                        accXv = thirdordButtFilt(accXv, 1/41.29, 0.02);
                        accYv = thirdordButtFilt(accYv, 1/41.29, 0.02);
                        accZv = thirdordButtFilt(accZv, 1/41.29, 0.02);

                        vector<float> body_accXv = butterworthFilter(accXv, 0.3, 2);
                        vector<float> body_accYv = butterworthFilter(accYv, 0.3, 2);
                        vector<float> body_accZv = butterworthFilter(accZv, 0.3, 2);

                        vector<float> grav_accXv = getGravity(accXv, body_accXv);
                        vector<float> grav_accYv = getGravity(accYv, body_accYv);
                        vector<float> grav_accZv = getGravity(accZv, body_accZv);
                        
                        // JERK
                        vector<float> velocity_x = calculate_velocity(gyroXv, timev);
                        vector<float> velocity_y = calculate_velocity(gyroYv, timev);
                        vector<float> velocity_z = calculate_velocity(gyroZv, timev);
                        
                        vector<float> j_gyroXv = calculate_jerk(velocity_x, timev);
                        vector<float> j_gyroYv = calculate_jerk(velocity_y, timev);
                        vector<float> j_gyroZv = calculate_jerk(velocity_z, timev);

                        velocity_x = calculate_velocity(body_accXv, timev);
                        velocity_y = calculate_velocity(body_accYv, timev);
                        velocity_z = calculate_velocity(body_accZv, timev);

                        vector<float> j_accXv = calculate_jerk(velocity_x, timev);
                        vector<float> j_accYv = calculate_jerk(velocity_y, timev);
                        vector<float> j_accZv = calculate_jerk(velocity_z, timev);

                        velocity_x.clear();
                        velocity_y.clear();
                        velocity_z.clear();

                        // MAG
                        vector<float> mag_body_acc_v = getMagVector(body_accXv, body_accYv, body_accZv);
                        vector<float> mag_grav_acc_v = getMagVector(grav_accXv, grav_accYv, grav_accZv);
                        vector<float> mag_body_j_acc_v = getMagVector(j_accXv, j_accYv, j_accZv);
                        vector<float> mag_gyro_v = getMagVector(gyroXv, gyroYv, gyroZv);
                        vector<float> mag_j_gyro_v = getMagVector(j_gyroXv, j_gyroYv, j_gyroZv);
                        // MEAN
                        float mean_accx = average(body_accXv);
                        float mean_accy = average(body_accYv);
                        float mean_accz = average(body_accZv);
                        float mean_g_accx = average(grav_accXv);
                        float mean_g_accy = average(grav_accYv);
                        float mean_g_accz = average(grav_accZv);
                        float mean_gyrox = average(gyroXv);
                        float mean_gyroy = average(gyroYv);
                        float mean_gyroz = average(gyroZv);
                        float mag_mean_body_acc = average(mag_body_acc_v);
                        float mag_mean_grav_acc = average(mag_grav_acc_v);
                        float mag_mean_gyro = average(mag_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("val : %f", mean_accx);
                        delay(2000); */

                        //JERK MEAN
                        float j_mean_accx = average(j_accXv);
                        float j_mean_accy = average(j_accYv);
                        float j_mean_accz = average(j_accZv);
                        float j_mean_gyrox = average(j_gyroXv);
                        float j_mean_gyroy = average(j_gyroYv);
                        float j_mean_gyroz = average(j_gyroZv);
                        float mag_j_mean_body_acc = average(mag_body_j_acc_v);
                        float mag_j_mean_gyro = average(mag_j_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("j mean : %f", j_mean_accx);
                        delay(2000); */

                        // STD DEV
                        float accx_std = std_dev(body_accXv, mean_accx);
                        float accy_std = std_dev(body_accYv, mean_accy);
                        float accz_std = std_dev(body_accZv, mean_accz);
                        float g_accx_std = std_dev(grav_accXv, mean_g_accx);
                        float g_accy_std = std_dev(grav_accYv, mean_g_accy);
                        float g_accz_std = std_dev(grav_accZv, mean_g_accz);
                        float gyrox_std = std_dev(gyroXv, mean_gyrox);
                        float gyroy_std = std_dev(gyroYv, mean_gyroy);
                        float gyroz_std = std_dev(gyroZv, mean_gyroz);
                        float mag_body_acc_std = std_dev(mag_body_acc_v, mag_mean_body_acc);
                        float mag_grav_acc_std = std_dev(mag_grav_acc_v, mag_mean_grav_acc);
                        float mag_gyro_std = std_dev(mag_gyro_v, mag_mean_gyro);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("std : %f", accx_std);
                        delay(2000); */

                        // JERK STD DEV
                        float j_accx_std = std_dev(j_accXv, j_mean_accx);
                        float j_accy_std = std_dev(j_accYv, j_mean_accy);
                        float j_accz_std = std_dev(j_accZv, j_mean_accz);
                        float j_gyrox_std = std_dev(j_gyroXv, j_mean_gyrox);
                        float j_gyroy_std = std_dev(j_gyroYv, j_mean_gyroy);
                        float j_gyroz_std = std_dev(j_gyroZv, j_mean_gyroz);
                        float mag_j_body_acc_std = std_dev(mag_body_j_acc_v, mag_j_mean_body_acc);
                        float mag_j_gyro_std = std_dev(mag_j_gyro_v, mag_j_mean_gyro);
                        //M5.Lcd.printf("std : %f", j_accx_std);
                        //delay(2000);

                        //MEDIAN ABSOLUTE DEVIATION
                        float accx_mad = mad(body_accXv);
                        float accy_mad = mad(body_accYv);
                        float accz_mad = mad(body_accZv);
                        float g_accx_mad = mad(grav_accXv);
                        float g_accy_mad = mad(grav_accYv);
                        float g_accz_mad = mad(grav_accZv);
                        float gyrox_mad = mad(gyroXv);
                        float gyroy_mad = mad(gyroYv);
                        float gyroz_mad = mad(gyroZv);
                        float mag_body_acc_mad = mad(mag_body_acc_v);
                        float mag_grav_acc_mad = mad(mag_grav_acc_v);
                        float mag_gyro_mad = mad(mag_gyro_v);
                        //M5.Lcd.printf("mad : %f", accx_mad);
                        //delay(2000);

                        // JERK MEDIAN ABSOLUTE DEVIATION
                        float j_accx_mad = mad(j_accXv);
                        float j_accy_mad = mad(j_accYv);
                        float j_accz_mad = mad(j_accZv);
                        float j_gyrox_mad = mad(j_gyroXv);
                        float j_gyroy_mad = mad(j_gyroYv);
                        float j_gyroz_mad = mad(j_gyroZv);
                        float mag_j_body_acc_mad = mad(mag_body_j_acc_v);
                        float mag_j_gyro_mad = mad(mag_j_gyro_v);
                        //M5.Lcd.printf("j mad : %f", j_accx_mad);
                        //delay(2000);

                        // TODO 

                        // MAX
                        float accx_max = max_v(body_accXv);
                        float accy_max = max_v(body_accYv);
                        float accz_max = max_v(body_accZv);
                        float g_accx_max = max_v(grav_accXv);
                        float g_accy_max = max_v(grav_accYv);
                        float g_accz_max = max_v(grav_accZv);
                        float gyrox_max = max_v(gyroXv);
                        float gyroy_max = max_v(gyroYv);
                        float gyroz_max = max_v(gyroZv);
                        float mag_body_acc_max = max_v(mag_body_acc_v);
                        float mag_grav_acc_max = max_v(mag_grav_acc_v);
                        float mag_gyro_max = max_v(mag_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("max : %f", g_accx_max);
                        delay(2000); */

                        // JERK MAX
                        float j_accx_max = max_v(j_accXv);
                        float j_accy_max = max_v(j_accYv);
                        float j_accz_max = max_v(j_accZv);
                        float j_gyrox_max = max_v(j_gyroXv);
                        float j_gyroy_max = max_v(j_gyroYv);
                        float j_gyroz_max = max_v(j_gyroZv);
                        float mag_j_body_acc_max = max_v(mag_body_j_acc_v);
                        float mag_j_gyro_max = max_v(mag_j_gyro_v);

                        // MIN
                        float accx_min = min_v(body_accXv);
                        float accy_min = min_v(body_accYv);
                        float accz_min = min_v(body_accZv);
                        float g_accx_min = min_v(grav_accXv);
                        float g_accy_min = min_v(grav_accYv);
                        float g_accz_min = min_v(grav_accZv);
                        float gyrox_min = min_v(gyroXv);
                        float gyroy_min = min_v(gyroYv);
                        float gyroz_min = min_v(gyroZv);
                        float mag_body_acc_min = min_v(mag_body_acc_v);
                        float mag_grav_acc_min = min_v(mag_grav_acc_v);
                        float mag_gyro_min = min_v(mag_gyro_v);

                        // JERK MIN
                        float j_accx_min = min_v(j_accXv);
                        float j_accy_min = min_v(j_accYv);
                        float j_accz_min = min_v(j_accZv);
                        float j_gyrox_min = min_v(j_gyroXv);
                        float j_gyroy_min = min_v(j_gyroYv);
                        float j_gyroz_min = min_v(j_gyroZv);
                        float mag_j_body_acc_min = min_v(mag_body_j_acc_v);
                        float mag_j_gyro_min = min_v(mag_j_gyro_v);

                        // SMA
                        float acc_sma = signalMagnitudeArea(body_accXv, body_accYv, body_accZv, 0, body_accXv.size());
                        float g_acc_sma = signalMagnitudeArea(grav_accXv, grav_accYv, grav_accZv, 0, grav_accXv.size());
                        float gyro_sma = signalMagnitudeArea(gyroXv, gyroYv, gyroZv, 0, gyroXv.size());
                        float mag_body_acc_sma = getSMAfromMag(mag_body_acc_v);
                        float mag_grav_acc_sma = getSMAfromMag(mag_grav_acc_v);
                        float mag_gyro_sma = getSMAfromMag(mag_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("sma : %f", acc_sma);
                        delay(2000); */

                        // JERK SMA
                        float j_acc_sma = signalMagnitudeArea(j_accXv, j_accYv, j_accZv, 0, j_accXv.size());
                        float j_gyro_sma = signalMagnitudeArea(j_gyroXv, j_gyroYv, j_gyroZv, 0, j_gyroXv.size());
                        float mag_j_body_acc_sma = getSMAfromMag(mag_body_j_acc_v);
                        float mag_j_gyro_sma = getSMAfromMag(mag_j_gyro_v);

                        // ENERGY
                        float accx_energy = signalEnergy(body_accXv, 0, body_accXv.size());
                        float accy_energy = signalEnergy(body_accYv, 0, body_accYv.size());
                        float accz_energy = signalEnergy(body_accZv, 0, body_accZv.size());
                        float g_accx_energy = signalEnergy(grav_accXv, 0, grav_accXv.size());
                        float g_accy_energy = signalEnergy(grav_accYv, 0, grav_accYv.size());
                        float g_accz_energy = signalEnergy(grav_accZv, 0, grav_accZv.size());
                        float gyrox_energy = signalEnergy(gyroXv, 0, gyroXv.size());
                        float gyroy_energy = signalEnergy(gyroYv, 0, gyroYv.size());
                        float gyroz_energy = signalEnergy(gyroZv, 0, gyroZv.size());
                        float mag_body_acc_energy = signalEnergy(mag_body_acc_v, 0, mag_body_acc_v.size());
                        float mag_grav_acc_energy = signalEnergy(mag_grav_acc_v, 0, mag_grav_acc_v.size());
                        float mag_gyro_energy = signalEnergy(mag_gyro_v, 0, mag_gyro_v.size());
                        

                        // JERK ENERGY
                        float j_accx_energy = signalEnergy(j_accXv, 0, j_accXv.size());
                        float j_accy_energy = signalEnergy(j_accYv, 0, j_accYv.size());
                        float j_accz_energy = signalEnergy(j_accZv, 0, j_accZv.size());
                        float j_gyrox_energy = signalEnergy(j_gyroXv, 0, j_gyroXv.size());
                        float j_gyroy_energy = signalEnergy(j_gyroYv, 0, j_gyroYv.size());
                        float j_gyroz_energy = signalEnergy(j_gyroZv, 0, j_gyroZv.size());
                        float mag_j_body_acc_energy = signalEnergy(mag_body_j_acc_v, 0, mag_body_j_acc_v.size());
                        float mag_j_gyro_energy = signalEnergy(mag_j_gyro_v, 0, mag_j_gyro_v.size());

                        // IQR
                        float accx_iqr = signalInterquartileRange(body_accXv);
                        float accy_iqr = signalInterquartileRange(body_accYv);
                        float accz_iqr = signalInterquartileRange(body_accZv);
                        float g_accx_iqr = signalInterquartileRange(grav_accXv);
                        float g_accy_iqr = signalInterquartileRange(grav_accYv);
                        float g_accz_iqr = signalInterquartileRange(grav_accZv);
                        float gyrox_iqr = signalInterquartileRange(gyroXv);
                        float gyroy_iqr = signalInterquartileRange(gyroYv);
                        float gyroz_iqr = signalInterquartileRange(gyroZv);
                        float mag_body_acc_iqr = signalInterquartileRange(mag_body_acc_v);
                        float mag_grav_acc_iqr = signalInterquartileRange(mag_grav_acc_v);
                        float mag_gyro_iqr = signalInterquartileRange(mag_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("accx d : %f", accx_iqr);
                        M5.Lcd.setCursor(0, 182);
                        M5.Lcd.printf("accy d : %f", g_accx_iqr);
                        M5.Lcd.setCursor(0, 202);
                        M5.Lcd.printf("accz d : %f", gyrox_iqr);
                        delay(2000); */
                        

                        // JERK IQR
                        float j_accx_iqr = signalInterquartileRange(j_accXv);
                        float j_accy_iqr = signalInterquartileRange(j_accYv);
                        float j_accz_iqr = signalInterquartileRange(j_accZv);
                        float j_gyrox_iqr = signalInterquartileRange(j_gyroXv);
                        float j_gyroy_iqr = signalInterquartileRange(j_gyroYv);
                        float j_gyroz_iqr = signalInterquartileRange(j_gyroZv);
                        float mag_j_body_acc_iqr = signalInterquartileRange(mag_body_j_acc_v);
                        float mag_j_gyro_iqr = signalInterquartileRange(mag_j_gyro_v);

                        // ENTROPY
                        float accx_entr = signalEntropy(body_accXv, 0, body_accXv.size());
                        float accy_entr = signalEntropy(body_accYv, 0, body_accYv.size());
                        float accz_entr = signalEntropy(body_accZv, 0, body_accZv.size());
                        float g_accx_entr = signalEntropy(grav_accXv, 0, grav_accXv.size());
                        float g_accy_entr = signalEntropy(grav_accYv, 0, grav_accYv.size());
                        float g_accz_entr = signalEntropy(grav_accZv, 0, grav_accZv.size());
                        float gyrox_entr = signalEntropy(gyroXv, 0, gyroXv.size());
                        float gyroy_entr = signalEntropy(gyroYv, 0, gyroYv.size());
                        float gyroz_entr = signalEntropy(gyroZv, 0, gyroZv.size());
                        float mag_body_acc_entr = signalEntropy(mag_body_acc_v, 0, mag_body_acc_v.size());
                        float mag_grav_acc_entr = signalEntropy(mag_grav_acc_v, 0, mag_grav_acc_v.size());
                        float mag_gyro_entr = signalEntropy(mag_gyro_v, 0, mag_gyro_v.size());

                        // JERK ENTROPY
                        float j_accx_entr = signalEntropy(j_accXv, 0, j_accXv.size());
                        float j_accy_entr = signalEntropy(j_accYv, 0, j_accYv.size());
                        float j_accz_entr = signalEntropy(j_accZv, 0, j_accZv.size());
                        float j_gyrox_entr = signalEntropy(j_gyroXv, 0, j_gyroXv.size());
                        float j_gyroy_entr = signalEntropy(j_gyroYv, 0, j_gyroYv.size());
                        float j_gyroz_entr = signalEntropy(j_gyroZv, 0, j_gyroZv.size());
                        float mag_j_body_acc_entr = signalEntropy(mag_body_j_acc_v, 0, mag_body_j_acc_v.size());
                        float mag_j_gyro_entr = signalEntropy(mag_j_gyro_v, 0, mag_j_gyro_v.size());

                        // AR COEFFICIENTS
                        int order = 4;
                        vector<float> accx_ar_c = signalAutoregressionCoefficientsBurg(body_accXv, order);
                        vector<float> accy_ar_c = signalAutoregressionCoefficientsBurg(body_accYv, order);
                        vector<float> accz_ar_c = signalAutoregressionCoefficientsBurg(body_accZv, order);
                        vector<float> g_accx_ar_c = signalAutoregressionCoefficientsBurg(grav_accXv, order);
                        vector<float> g_accy_ar_c = signalAutoregressionCoefficientsBurg(grav_accYv, order);
                        vector<float> g_accz_ar_c = signalAutoregressionCoefficientsBurg(grav_accZv, order);
                        vector<float> gyrox_ar_c = signalAutoregressionCoefficientsBurg(gyroXv, order);
                        vector<float> gyroy_ar_c = signalAutoregressionCoefficientsBurg(gyroYv, order);
                        vector<float> gyroz_ar_c = signalAutoregressionCoefficientsBurg(gyroZv, order);
                        vector<float> mag_body_acc_ar_c = signalAutoregressionCoefficientsBurg(mag_body_acc_v, order);
                        vector<float> mag_grav_acc_ar_c = signalAutoregressionCoefficientsBurg(mag_grav_acc_v, order);
                        vector<float> mag_gyro_ar_c = signalAutoregressionCoefficientsBurg(mag_gyro_v, order);

                        // JERK AR COEFFICIENTS
                        vector<float> j_accx_ar_c = signalAutoregressionCoefficientsBurg(j_accXv, order);
                        vector<float> j_accy_ar_c = signalAutoregressionCoefficientsBurg(j_accYv, order);
                        vector<float> j_accz_ar_c = signalAutoregressionCoefficientsBurg(j_accZv, order);
                        vector<float> j_gyrox_ar_c = signalAutoregressionCoefficientsBurg(j_gyroXv, order);
                        vector<float> j_gyroy_ar_c = signalAutoregressionCoefficientsBurg(j_gyroYv, order);
                        vector<float> j_gyroz_ar_c = signalAutoregressionCoefficientsBurg(j_gyroZv, order);
                        vector<float> mag_j_body_acc_ar_c = signalAutoregressionCoefficientsBurg(mag_body_j_acc_v, order);
                        vector<float> mag_j_gyro_ar_c = signalAutoregressionCoefficientsBurg(mag_j_gyro_v, order);

                        // CORRELATION
                        float accxy_corr = correlationCoefficient(body_accXv, body_accYv);
                        float accxz_corr = correlationCoefficient(body_accXv, body_accZv);
                        float accyz_corr = correlationCoefficient(body_accYv, body_accZv);
                        float g_accxy_corr = correlationCoefficient(grav_accXv, grav_accYv);
                        float g_accxz_corr = correlationCoefficient(grav_accXv, grav_accZv);
                        float g_accyz_corr = correlationCoefficient(grav_accYv, grav_accZv);
                        float gyroxy_corr = correlationCoefficient(gyroXv, gyroYv);
                        float gyroxz_corr = correlationCoefficient(gyroXv, gyroZv);
                        float gyroyz_corr = correlationCoefficient(gyroYv, gyroZv);

                        // JERK CORRELATION
                        float j_accxy_corr = correlationCoefficient(j_accXv, j_accYv);
                        float j_accxz_corr = correlationCoefficient(j_accXv, j_accZv);
                        float j_accyz_corr = correlationCoefficient(j_accYv, j_accZv);
                        float j_gyroxy_corr = correlationCoefficient(j_gyroXv, j_gyroYv);
                        float j_gyroxz_corr = correlationCoefficient(j_gyroXv, j_gyroZv);
                        float j_gyroyz_corr = correlationCoefficient(j_gyroYv, j_gyroZv);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("jerk corr : %f", j_gyroyz_corr);
                        delay(2000); */

                        gyroXv.clear();
                        gyroYv.clear();
                        gyroZv.clear();
                        j_gyroXv.clear();
                        j_gyroYv.clear();
                        j_gyroZv.clear();
                        accXv.clear();
                        accYv.clear();                  
                        accZv.clear();
                        j_accXv.clear();
                        j_accYv.clear();                  
                        j_accZv.clear();
                        body_accXv.clear();
                        body_accYv.clear();
                        body_accZv.clear();
                        grav_accXv.clear();
                        grav_accYv.clear();
                        grav_accZv.clear();
                        mag_body_acc_v.clear();
                        mag_grav_acc_v.clear();
                        mag_body_j_acc_v.clear();
                        mag_gyro_v.clear();
                        mag_j_gyro_v.clear();

                        features.push_back(mean_accx);
                        features.push_back(mean_accy);
                        features.push_back(accx_std);
                        features.push_back(accy_std);
                        features.push_back(accz_std);
                        features.push_back(accx_mad);
                        features.push_back(accy_mad);
                        features.push_back(accz_mad);
                        features.push_back(accx_max);
                        features.push_back(accy_max);
                        features.push_back(accz_max);
                        features.push_back(accx_min);
                        features.push_back(accy_min);
                        features.push_back(accz_min);
                        features.push_back(acc_sma);
                        features.push_back(accx_energy);
                        features.push_back(accx_iqr);
                        features.push_back(accy_iqr);
                        features.push_back(accz_iqr);
                        features.push_back(accx_entr);
                        features.push_back(accy_entr);
                        features.push_back(accz_entr);
                        for (size_t i = 0; i < accx_ar_c.size(); i++)
                        {
                            /* M5.Lcd.setCursor(0, 222);
                            M5.Lcd.printf("accx_ar_c %d: %f", i, accx_ar_c[i]);
                            delay(2000); */
                            features.push_back(accx_ar_c[i]);
                        }
                        for (size_t i = 0; i < accy_ar_c.size(); i++)
                        {
                            features.push_back(accy_ar_c[i]);
                        }
                        for (size_t i = 0; i < accz_ar_c.size(); i++)
                        {
                            features.push_back(accz_ar_c[i]);
                        }
                        features.push_back(accxy_corr);
                        features.push_back(accxz_corr);
                        features.push_back(accyz_corr);
                        //pushing gravity
                        features.push_back(mean_g_accx);
                        features.push_back(mean_g_accy);
                        features.push_back(mean_g_accz);
                        features.push_back(g_accx_max);
                        features.push_back(g_accy_max);
                        features.push_back(g_accz_max);
                        features.push_back(g_accx_min);
                        features.push_back(g_accy_min);
                        features.push_back(g_accz_min);
                        features.push_back(g_acc_sma);
                        features.push_back(g_accx_energy);
                        features.push_back(g_accy_energy);
                        features.push_back(g_accz_energy);
                        features.push_back(g_accx_entr);
                        for (size_t i = 1; i < g_accx_ar_c.size(); i = i + 2)
                        {
                            features.push_back(g_accx_ar_c[i]);
                        }
                        for (size_t i = 1; i < g_accy_ar_c.size(); i = i + 2)
                        {
                            features.push_back(g_accy_ar_c[i]);
                        }
                        for (size_t i = 1; i < g_accz_ar_c.size(); i = i + 2)
                        {
                            features.push_back(g_accz_ar_c[i]);
                        }
                        features.push_back(g_accxy_corr);
                        features.push_back(g_accxz_corr);
                        features.push_back(g_accyz_corr);
                        // pushing Acc Jerk
                        features.push_back(j_mean_accx);
                        features.push_back(j_mean_accy);
                        features.push_back(j_mean_accz);
                        features.push_back(j_accx_std);
                        features.push_back(j_accy_std);
                        features.push_back(j_accz_std);
                        features.push_back(j_accx_mad);
                        features.push_back(j_accy_mad);
                        features.push_back(j_accz_mad);
                        features.push_back(j_accx_max);
                        features.push_back(j_accy_max);
                        features.push_back(j_accz_max);
                        features.push_back(j_accx_min);
                        features.push_back(j_accy_min);
                        features.push_back(j_accz_min);
                        features.push_back(j_acc_sma);
                        features.push_back(j_accx_energy);
                        features.push_back(j_accy_energy);
                        features.push_back(j_accx_iqr);
                        features.push_back(j_accy_iqr);
                        features.push_back(j_accz_iqr);
                        features.push_back(j_accx_entr);
                        features.push_back(j_accy_entr);
                        features.push_back(j_accz_entr);
                        for (size_t i = 0; i < j_accx_ar_c.size(); i++)
                        {
                            features.push_back(j_accx_ar_c[i]);
                        }
                        for (size_t i = 0; i < j_accy_ar_c.size(); i++)
                        {
                            features.push_back(j_accy_ar_c[i]);
                        }
                        for (size_t i = 0; i < j_accz_ar_c.size(); i++)
                        {
                            features.push_back(j_accz_ar_c[i]);
                        }
                        features.push_back(j_accxz_corr);
                        features.push_back(j_accyz_corr);
                        //pushing Gyro
                        features.push_back(mean_gyrox);
                        features.push_back(mean_gyroy);
                        features.push_back(mean_gyroz);
                        features.push_back(gyrox_std);
                        features.push_back(gyroy_std);
                        features.push_back(gyroz_std);
                        features.push_back(gyrox_mad);
                        features.push_back(gyroy_mad);
                        features.push_back(gyroz_mad);
                        features.push_back(gyrox_max);
                        features.push_back(gyroz_max);
                        features.push_back(gyrox_min);
                        features.push_back(gyroy_min);
                        features.push_back(gyroz_min);
                        features.push_back(gyro_sma);
                        features.push_back(gyrox_iqr);
                        features.push_back(gyroy_iqr);
                        features.push_back(gyroz_iqr);
                        features.push_back(gyrox_entr);
                        features.push_back(gyroy_entr);
                        features.push_back(gyroz_entr);
                        for (size_t i = 1; i < gyrox_ar_c.size(); i++)
                        {
                            features.push_back(gyrox_ar_c[i]);
                        }
                        for (size_t i = 1; i < gyroy_ar_c.size(); i++)
                        {
                            features.push_back(gyroy_ar_c[i]);
                        }
                        for (size_t i = 0; i < gyroz_ar_c.size(); i++)
                        {
                            features.push_back(gyroz_ar_c[i]);
                        }
                        features.push_back(gyroxy_corr);
                        features.push_back(gyroxz_corr);
                        features.push_back(gyroyz_corr);
                        // pushing Gyro Jerk
                        //features.push_back(j_mean_gyrox);
                        features.push_back(j_mean_gyroy);
                        features.push_back(j_mean_gyroz);
                        features.push_back(j_gyrox_std);
                        features.push_back(j_gyroy_std);
                        features.push_back(j_gyroz_std);
                        features.push_back(j_gyrox_mad);
                        features.push_back(j_gyroy_mad);
                        features.push_back(j_gyroz_mad);
                        features.push_back(j_gyrox_max);
                        features.push_back(j_gyroz_max);
                        features.push_back(j_gyrox_min);
                        features.push_back(j_gyroy_min);
                        features.push_back(j_gyroz_min);
                        features.push_back(j_gyro_sma);
                        features.push_back(j_gyrox_iqr);
                        features.push_back(j_gyroz_iqr);
                        features.push_back(j_gyrox_entr);
                        features.push_back(j_gyroy_entr);
                        features.push_back(j_gyroz_entr);
                        for (size_t i = 0; i < j_gyrox_ar_c.size(); i++)
                        {
                            features.push_back(j_gyrox_ar_c[i]);
                        }
                        for (size_t i = 1; i < j_gyroy_ar_c.size(); i++)
                        {
                            features.push_back(j_gyroy_ar_c[i]);
                        }
                        for (size_t i = 0; i < j_gyroz_ar_c.size(); i++)
                        {
                            features.push_back(j_gyroz_ar_c[i]);
                        }
                        features.push_back(j_gyroxy_corr);
                        features.push_back(j_gyroxz_corr);
                        features.push_back(j_gyroyz_corr);
                        // pushing Acc Mag
                        features.push_back(mag_mean_body_acc);
                        features.push_back(mag_body_acc_std);
                        features.push_back(mag_body_acc_mad);
                        features.push_back(mag_body_acc_max);
                        features.push_back(mag_body_acc_sma);
                        features.push_back(mag_body_acc_energy);
                        features.push_back(mag_body_acc_iqr);
                        features.push_back(mag_body_acc_entr);
                        for (size_t i = 0; i < mag_body_acc_ar_c.size(); i++)
                        {
                            features.push_back(mag_body_acc_ar_c[i]);
                        }
                        // pushing Grav Mag
                        features.push_back(mag_mean_grav_acc);
                        features.push_back(mag_grav_acc_std);
                        features.push_back(mag_grav_acc_mad);
                        features.push_back(mag_grav_acc_max);
                        features.push_back(mag_grav_acc_sma);
                        features.push_back(mag_grav_acc_energy);
                        features.push_back(mag_grav_acc_iqr);
                        features.push_back(mag_grav_acc_entr);
                        for (size_t i = 0; i < mag_grav_acc_ar_c.size(); i++)
                        {
                            features.push_back(mag_grav_acc_ar_c[i]);
                        }
                        // pushing Acc Jerk Mag
                        features.push_back(mag_j_mean_body_acc);
                        features.push_back(mag_j_body_acc_std);
                        features.push_back(mag_j_body_acc_mad);
                        features.push_back(mag_j_body_acc_max);
                        features.push_back(mag_j_body_acc_sma);
                        features.push_back(mag_j_body_acc_energy);
                        features.push_back(mag_j_body_acc_iqr);
                        features.push_back(mag_j_body_acc_entr);
                        for (size_t i = 0; i < mag_j_body_acc_ar_c.size(); i++)
                        {
                            features.push_back(mag_j_body_acc_ar_c[i]);
                        }
                        // pushing Gyro Mag
                        features.push_back(mag_mean_gyro);
                        features.push_back(mag_gyro_std);
                        features.push_back(mag_gyro_mad);
                        features.push_back(mag_gyro_max);
                        features.push_back(mag_gyro_min);
                        features.push_back(mag_gyro_sma);
                        features.push_back(mag_gyro_iqr);
                        features.push_back(mag_gyro_entr);
                        for (size_t i = 0; i < mag_gyro_ar_c.size(); i++)
                        {
                            features.push_back(mag_gyro_ar_c[i]);
                        }
                        // pushing Gyro Jerk Mag
                        features.push_back(mag_j_mean_gyro);
                        features.push_back(mag_j_gyro_std);
                        features.push_back(mag_j_gyro_mad);
                        features.push_back(mag_j_gyro_max);
                        features.push_back(mag_j_gyro_sma);
                        features.push_back(mag_j_gyro_iqr);
                        features.push_back(mag_j_gyro_entr);
                        for (size_t i = 0; i < mag_j_gyro_ar_c.size(); i++)
                        {
                            if (i != 1){
                                features.push_back(mag_j_gyro_ar_c[i]);
                            }
                        }
                    }
                    
                    // SOM
                    if (act_count == 8){

                        /* M5.Lcd.clear();
                        M5.Lcd.setCursor(0, 222);
                        M5.Lcd.printf("gyroXv: %f", gyroXv[0]);
                        delay(2000); */
                        gyroXv = medianFilter(gyroXv, gyroXv.size() / 2);
                        gyroYv = medianFilter(gyroYv, gyroYv.size() / 2);
                        gyroZv = medianFilter(gyroZv, gyroZv.size() / 2);
                        accXv = medianFilter(accXv, accXv.size() / 2);
                        accYv = medianFilter(accYv, accYv.size() / 2);
                        accZv = medianFilter(accZv, accZv.size() / 2);

                        /* M5.Lcd.clear();
                        M5.Lcd.setCursor(0, 222);
                        M5.Lcd.printf("gyroXv d_mf: %f", gyroXv[0]);
                        delay(2000); */
                        
                        gyroXv = thirdordButtFilt(gyroXv, 1/41.29, 0.02);
                        gyroYv = thirdordButtFilt(gyroYv, 1/41.29, 0.02);
                        gyroZv = thirdordButtFilt(gyroZv, 1/41.29, 0.02);
                        accXv = thirdordButtFilt(accXv, 1/41.29, 0.02);
                        accYv = thirdordButtFilt(accYv, 1/41.29, 0.02);
                        accZv = thirdordButtFilt(accZv, 1/41.29, 0.02);

                        /* M5.Lcd.clear();
                        M5.Lcd.setCursor(0, 222);
                        M5.Lcd.printf("gyroXv d_bf: %f", gyroXv[0]);
                        delay(2000); */

                        vector<float> body_accXv = butterworthFilter(accXv, 0.3, 2);
                        vector<float> body_accYv = butterworthFilter(accYv, 0.3, 2);
                        vector<float> body_accZv = butterworthFilter(accZv, 0.3, 2);

                        vector<float> grav_accXv = getGravity(accXv, body_accXv);
                        vector<float> grav_accYv = getGravity(accYv, body_accYv);
                        vector<float> grav_accZv = getGravity(accZv, body_accZv);
                        
                        // JERK
                        vector<float> velocity_x = calculate_velocity(gyroXv, timev);
                        vector<float> velocity_y = calculate_velocity(gyroYv, timev);
                        vector<float> velocity_z = calculate_velocity(gyroZv, timev);
                        
                        vector<float> j_gyroXv = calculate_jerk(velocity_x, timev);
                        vector<float> j_gyroYv = calculate_jerk(velocity_y, timev);
                        vector<float> j_gyroZv = calculate_jerk(velocity_z, timev);

                        velocity_x = calculate_velocity(body_accXv, timev);
                        velocity_y = calculate_velocity(body_accYv, timev);
                        velocity_z = calculate_velocity(body_accZv, timev);

                        vector<float> j_accXv = calculate_jerk(velocity_x, timev);
                        vector<float> j_accYv = calculate_jerk(velocity_y, timev);
                        vector<float> j_accZv = calculate_jerk(velocity_z, timev);

                        velocity_x.clear();
                        velocity_y.clear();
                        velocity_z.clear();

                        // MAG
                        vector<float> mag_body_acc_v = getMagVector(body_accXv, body_accYv, body_accZv);
                        vector<float> mag_grav_acc_v = getMagVector(grav_accXv, grav_accYv, grav_accZv);
                        vector<float> mag_body_j_acc_v = getMagVector(j_accXv, j_accYv, j_accZv);
                        vector<float> mag_gyro_v = getMagVector(gyroXv, gyroYv, gyroZv);
                        vector<float> mag_j_gyro_v = getMagVector(j_gyroXv, j_gyroYv, j_gyroZv);
                        // MEAN
                        float mean_accx = average(body_accXv);
                        float mean_accy = average(body_accYv);
                        float mean_accz = average(body_accZv);
                        float mean_g_accx = average(grav_accXv);
                        float mean_g_accy = average(grav_accYv);
                        float mean_g_accz = average(grav_accZv);
                        float mean_gyrox = average(gyroXv);
                        float mean_gyroy = average(gyroYv);
                        float mean_gyroz = average(gyroZv);
                        float mag_mean_body_acc = average(mag_body_acc_v);
                        float mag_mean_grav_acc = average(mag_grav_acc_v);
                        float mag_mean_gyro = average(mag_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("val : %f", mean_accx);
                        delay(2000); */

                        //JERK MEAN
                        float j_mean_accx = average(j_accXv);
                        float j_mean_accy = average(j_accYv);
                        float j_mean_accz = average(j_accZv);
                        float j_mean_gyrox = average(j_gyroXv);
                        float j_mean_gyroy = average(j_gyroYv);
                        float j_mean_gyroz = average(j_gyroZv);
                        float mag_j_mean_body_acc = average(mag_body_j_acc_v);
                        float mag_j_mean_gyro = average(mag_j_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("j mean : %f", j_mean_accx);
                        delay(2000); */

                        // STD DEV
                        float accx_std = std_dev(body_accXv, mean_accx);
                        float accy_std = std_dev(body_accYv, mean_accy);
                        float accz_std = std_dev(body_accZv, mean_accz);
                        float g_accx_std = std_dev(grav_accXv, mean_g_accx);
                        float g_accy_std = std_dev(grav_accYv, mean_g_accy);
                        float g_accz_std = std_dev(grav_accZv, mean_g_accz);
                        float gyrox_std = std_dev(gyroXv, mean_gyrox);
                        float gyroy_std = std_dev(gyroYv, mean_gyroy);
                        float gyroz_std = std_dev(gyroZv, mean_gyroz);
                        float mag_body_acc_std = std_dev(mag_body_acc_v, mag_mean_body_acc);
                        float mag_grav_acc_std = std_dev(mag_grav_acc_v, mag_mean_grav_acc);
                        float mag_gyro_std = std_dev(mag_gyro_v, mag_mean_gyro);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("std : %f", accx_std);
                        delay(2000); */

                        // JERK STD DEV
                        float j_accx_std = std_dev(j_accXv, j_mean_accx);
                        float j_accy_std = std_dev(j_accYv, j_mean_accy);
                        float j_accz_std = std_dev(j_accZv, j_mean_accz);
                        float j_gyrox_std = std_dev(j_gyroXv, j_mean_gyrox);
                        float j_gyroy_std = std_dev(j_gyroYv, j_mean_gyroy);
                        float j_gyroz_std = std_dev(j_gyroZv, j_mean_gyroz);
                        float mag_j_body_acc_std = std_dev(mag_body_j_acc_v, mag_j_mean_body_acc);
                        float mag_j_gyro_std = std_dev(mag_j_gyro_v, mag_j_mean_gyro);
                        //M5.Lcd.printf("std : %f", j_accx_std);
                        //delay(2000);

                        //MEDIAN ABSOLUTE DEVIATION
                        float accx_mad = mad(body_accXv);
                        float accy_mad = mad(body_accYv);
                        float accz_mad = mad(body_accZv);
                        float g_accx_mad = mad(grav_accXv);
                        float g_accy_mad = mad(grav_accYv);
                        float g_accz_mad = mad(grav_accZv);
                        float gyrox_mad = mad(gyroXv);
                        float gyroy_mad = mad(gyroYv);
                        float gyroz_mad = mad(gyroZv);
                        float mag_body_acc_mad = mad(mag_body_acc_v);
                        float mag_grav_acc_mad = mad(mag_grav_acc_v);
                        float mag_gyro_mad = mad(mag_gyro_v);
                        //M5.Lcd.printf("mad : %f", accx_mad);
                        //delay(2000);

                        // JERK MEDIAN ABSOLUTE DEVIATION
                        float j_accx_mad = mad(j_accXv);
                        float j_accy_mad = mad(j_accYv);
                        float j_accz_mad = mad(j_accZv);
                        float j_gyrox_mad = mad(j_gyroXv);
                        float j_gyroy_mad = mad(j_gyroYv);
                        float j_gyroz_mad = mad(j_gyroZv);
                        float mag_j_body_acc_mad = mad(mag_body_j_acc_v);
                        float mag_j_gyro_mad = mad(mag_j_gyro_v);
                        //M5.Lcd.printf("j mad : %f", j_accx_mad);
                        //delay(2000);

                        // TODO 

                        // MAX
                        float accx_max = max_v(body_accXv);
                        float accy_max = max_v(body_accYv);
                        float accz_max = max_v(body_accZv);
                        float g_accx_max = max_v(grav_accXv);
                        float g_accy_max = max_v(grav_accYv);
                        float g_accz_max = max_v(grav_accZv);
                        float gyrox_max = max_v(gyroXv);
                        float gyroy_max = max_v(gyroYv);
                        float gyroz_max = max_v(gyroZv);
                        float mag_body_acc_max = max_v(mag_body_acc_v);
                        float mag_grav_acc_max = max_v(mag_grav_acc_v);
                        float mag_gyro_max = max_v(mag_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("max : %f", g_accx_max);
                        delay(2000); */

                        // JERK MAX
                        float j_accx_max = max_v(j_accXv);
                        float j_accy_max = max_v(j_accYv);
                        float j_accz_max = max_v(j_accZv);
                        float j_gyrox_max = max_v(j_gyroXv);
                        float j_gyroy_max = max_v(j_gyroYv);
                        float j_gyroz_max = max_v(j_gyroZv);
                        float mag_j_body_acc_max = max_v(mag_body_j_acc_v);
                        float mag_j_gyro_max = max_v(mag_j_gyro_v);

                        // MIN
                        float accx_min = min_v(body_accXv);
                        float accy_min = min_v(body_accYv);
                        float accz_min = min_v(body_accZv);
                        float g_accx_min = min_v(grav_accXv);
                        float g_accy_min = min_v(grav_accYv);
                        float g_accz_min = min_v(grav_accZv);
                        float gyrox_min = min_v(gyroXv);
                        float gyroy_min = min_v(gyroYv);
                        float gyroz_min = min_v(gyroZv);
                        float mag_body_acc_min = min_v(mag_body_acc_v);
                        float mag_grav_acc_min = min_v(mag_grav_acc_v);
                        float mag_gyro_min = min_v(mag_gyro_v);

                        // JERK MIN
                        float j_accx_min = min_v(j_accXv);
                        float j_accy_min = min_v(j_accYv);
                        float j_accz_min = min_v(j_accZv);
                        float j_gyrox_min = min_v(j_gyroXv);
                        float j_gyroy_min = min_v(j_gyroYv);
                        float j_gyroz_min = min_v(j_gyroZv);
                        float mag_j_body_acc_min = min_v(mag_body_j_acc_v);
                        float mag_j_gyro_min = min_v(mag_j_gyro_v);

                        // SMA
                        float acc_sma = signalMagnitudeArea(body_accXv, body_accYv, body_accZv, 0, body_accXv.size());
                        float g_acc_sma = signalMagnitudeArea(grav_accXv, grav_accYv, grav_accZv, 0, grav_accXv.size());
                        float gyro_sma = signalMagnitudeArea(gyroXv, gyroYv, gyroZv, 0, gyroXv.size());
                        float mag_body_acc_sma = getSMAfromMag(mag_body_acc_v);
                        float mag_grav_acc_sma = getSMAfromMag(mag_grav_acc_v);
                        float mag_gyro_sma = getSMAfromMag(mag_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("sma : %f", acc_sma);
                        delay(2000); */

                        // JERK SMA
                        float j_acc_sma = signalMagnitudeArea(j_accXv, j_accYv, j_accZv, 0, j_accXv.size());
                        float j_gyro_sma = signalMagnitudeArea(j_gyroXv, j_gyroYv, j_gyroZv, 0, j_gyroXv.size());
                        float mag_j_body_acc_sma = getSMAfromMag(mag_body_j_acc_v);
                        float mag_j_gyro_sma = getSMAfromMag(mag_j_gyro_v);

                        // ENERGY
                        float accx_energy = signalEnergy(body_accXv, 0, body_accXv.size());
                        float accy_energy = signalEnergy(body_accYv, 0, body_accYv.size());
                        float accz_energy = signalEnergy(body_accZv, 0, body_accZv.size());
                        float g_accx_energy = signalEnergy(grav_accXv, 0, grav_accXv.size());
                        float g_accy_energy = signalEnergy(grav_accYv, 0, grav_accYv.size());
                        float g_accz_energy = signalEnergy(grav_accZv, 0, grav_accZv.size());
                        float gyrox_energy = signalEnergy(gyroXv, 0, gyroXv.size());
                        float gyroy_energy = signalEnergy(gyroYv, 0, gyroYv.size());
                        float gyroz_energy = signalEnergy(gyroZv, 0, gyroZv.size());
                        float mag_body_acc_energy = signalEnergy(mag_body_acc_v, 0, mag_body_acc_v.size());
                        float mag_grav_acc_energy = signalEnergy(mag_grav_acc_v, 0, mag_grav_acc_v.size());
                        float mag_gyro_energy = signalEnergy(mag_gyro_v, 0, mag_gyro_v.size());
                        

                        // JERK ENERGY
                        float j_accx_energy = signalEnergy(j_accXv, 0, j_accXv.size());
                        float j_accy_energy = signalEnergy(j_accYv, 0, j_accYv.size());
                        float j_accz_energy = signalEnergy(j_accZv, 0, j_accZv.size());
                        float j_gyrox_energy = signalEnergy(j_gyroXv, 0, j_gyroXv.size());
                        float j_gyroy_energy = signalEnergy(j_gyroYv, 0, j_gyroYv.size());
                        float j_gyroz_energy = signalEnergy(j_gyroZv, 0, j_gyroZv.size());
                        float mag_j_body_acc_energy = signalEnergy(mag_body_j_acc_v, 0, mag_body_j_acc_v.size());
                        float mag_j_gyro_energy = signalEnergy(mag_j_gyro_v, 0, mag_j_gyro_v.size());

                        // IQR
                        float accx_iqr = signalInterquartileRange(body_accXv);
                        float accy_iqr = signalInterquartileRange(body_accYv);
                        float accz_iqr = signalInterquartileRange(body_accZv);
                        float g_accx_iqr = signalInterquartileRange(grav_accXv);
                        float g_accy_iqr = signalInterquartileRange(grav_accYv);
                        float g_accz_iqr = signalInterquartileRange(grav_accZv);
                        float gyrox_iqr = signalInterquartileRange(gyroXv);
                        float gyroy_iqr = signalInterquartileRange(gyroYv);
                        float gyroz_iqr = signalInterquartileRange(gyroZv);
                        float mag_body_acc_iqr = signalInterquartileRange(mag_body_acc_v);
                        float mag_grav_acc_iqr = signalInterquartileRange(mag_grav_acc_v);
                        float mag_gyro_iqr = signalInterquartileRange(mag_gyro_v);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("accx d : %f", accx_iqr);
                        M5.Lcd.setCursor(0, 182);
                        M5.Lcd.printf("accy d : %f", g_accx_iqr);
                        M5.Lcd.setCursor(0, 202);
                        M5.Lcd.printf("accz d : %f", gyrox_iqr);
                        delay(2000); */
                        

                        // JERK IQR
                        float j_accx_iqr = signalInterquartileRange(j_accXv);
                        float j_accy_iqr = signalInterquartileRange(j_accYv);
                        float j_accz_iqr = signalInterquartileRange(j_accZv);
                        float j_gyrox_iqr = signalInterquartileRange(j_gyroXv);
                        float j_gyroy_iqr = signalInterquartileRange(j_gyroYv);
                        float j_gyroz_iqr = signalInterquartileRange(j_gyroZv);
                        float mag_j_body_acc_iqr = signalInterquartileRange(mag_body_j_acc_v);
                        float mag_j_gyro_iqr = signalInterquartileRange(mag_j_gyro_v);

                        // ENTROPY
                        float accx_entr = signalEntropy(body_accXv, 0, body_accXv.size());
                        float accy_entr = signalEntropy(body_accYv, 0, body_accYv.size());
                        float accz_entr = signalEntropy(body_accZv, 0, body_accZv.size());
                        float g_accx_entr = signalEntropy(grav_accXv, 0, grav_accXv.size());
                        float g_accy_entr = signalEntropy(grav_accYv, 0, grav_accYv.size());
                        float g_accz_entr = signalEntropy(grav_accZv, 0, grav_accZv.size());
                        float gyrox_entr = signalEntropy(gyroXv, 0, gyroXv.size());
                        float gyroy_entr = signalEntropy(gyroYv, 0, gyroYv.size());
                        float gyroz_entr = signalEntropy(gyroZv, 0, gyroZv.size());
                        float mag_body_acc_entr = signalEntropy(mag_body_acc_v, 0, mag_body_acc_v.size());
                        float mag_grav_acc_entr = signalEntropy(mag_grav_acc_v, 0, mag_grav_acc_v.size());
                        float mag_gyro_entr = signalEntropy(mag_gyro_v, 0, mag_gyro_v.size());

                        // JERK ENTROPY
                        float j_accx_entr = signalEntropy(j_accXv, 0, j_accXv.size());
                        float j_accy_entr = signalEntropy(j_accYv, 0, j_accYv.size());
                        float j_accz_entr = signalEntropy(j_accZv, 0, j_accZv.size());
                        float j_gyrox_entr = signalEntropy(j_gyroXv, 0, j_gyroXv.size());
                        float j_gyroy_entr = signalEntropy(j_gyroYv, 0, j_gyroYv.size());
                        float j_gyroz_entr = signalEntropy(j_gyroZv, 0, j_gyroZv.size());
                        float mag_j_body_acc_entr = signalEntropy(mag_body_j_acc_v, 0, mag_body_j_acc_v.size());
                        float mag_j_gyro_entr = signalEntropy(mag_j_gyro_v, 0, mag_j_gyro_v.size());

                        // AR COEFFICIENTS
                        int order = 4;
                        vector<float> accx_ar_c = signalAutoregressionCoefficientsBurg(body_accXv, order);
                        vector<float> accy_ar_c = signalAutoregressionCoefficientsBurg(body_accYv, order);
                        vector<float> accz_ar_c = signalAutoregressionCoefficientsBurg(body_accZv, order);
                        vector<float> g_accx_ar_c = signalAutoregressionCoefficientsBurg(grav_accXv, order);
                        vector<float> g_accy_ar_c = signalAutoregressionCoefficientsBurg(grav_accYv, order);
                        vector<float> g_accz_ar_c = signalAutoregressionCoefficientsBurg(grav_accZv, order);
                        vector<float> gyrox_ar_c = signalAutoregressionCoefficientsBurg(gyroXv, order);
                        vector<float> gyroy_ar_c = signalAutoregressionCoefficientsBurg(gyroYv, order);
                        vector<float> gyroz_ar_c = signalAutoregressionCoefficientsBurg(gyroZv, order);
                        vector<float> mag_body_acc_ar_c = signalAutoregressionCoefficientsBurg(mag_body_acc_v, order);
                        vector<float> mag_grav_acc_ar_c = signalAutoregressionCoefficientsBurg(mag_grav_acc_v, order);
                        vector<float> mag_gyro_ar_c = signalAutoregressionCoefficientsBurg(mag_gyro_v, order);

                        // JERK AR COEFFICIENTS
                        vector<float> j_accx_ar_c = signalAutoregressionCoefficientsBurg(j_accXv, order);
                        vector<float> j_accy_ar_c = signalAutoregressionCoefficientsBurg(j_accYv, order);
                        vector<float> j_accz_ar_c = signalAutoregressionCoefficientsBurg(j_accZv, order);
                        vector<float> j_gyrox_ar_c = signalAutoregressionCoefficientsBurg(j_gyroXv, order);
                        vector<float> j_gyroy_ar_c = signalAutoregressionCoefficientsBurg(j_gyroYv, order);
                        vector<float> j_gyroz_ar_c = signalAutoregressionCoefficientsBurg(j_gyroZv, order);
                        vector<float> mag_j_body_acc_ar_c = signalAutoregressionCoefficientsBurg(mag_body_j_acc_v, order);
                        vector<float> mag_j_gyro_ar_c = signalAutoregressionCoefficientsBurg(mag_j_gyro_v, order);

                        // CORRELATION
                        float accxy_corr = correlationCoefficient(body_accXv, body_accYv);
                        float accxz_corr = correlationCoefficient(body_accXv, body_accZv);
                        float accyz_corr = correlationCoefficient(body_accYv, body_accZv);
                        float g_accxy_corr = correlationCoefficient(grav_accXv, grav_accYv);
                        float g_accxz_corr = correlationCoefficient(grav_accXv, grav_accZv);
                        float g_accyz_corr = correlationCoefficient(grav_accYv, grav_accZv);
                        float gyroxy_corr = correlationCoefficient(gyroXv, gyroYv);
                        float gyroxz_corr = correlationCoefficient(gyroXv, gyroZv);
                        float gyroyz_corr = correlationCoefficient(gyroYv, gyroZv);

                        // JERK CORRELATION
                        float j_accxy_corr = correlationCoefficient(j_accXv, j_accYv);
                        float j_accxz_corr = correlationCoefficient(j_accXv, j_accZv);
                        float j_accyz_corr = correlationCoefficient(j_accYv, j_accZv);
                        float j_gyroxy_corr = correlationCoefficient(j_gyroXv, j_gyroYv);
                        float j_gyroxz_corr = correlationCoefficient(j_gyroXv, j_gyroZv);
                        float j_gyroyz_corr = correlationCoefficient(j_gyroYv, j_gyroZv);
                        /* M5.Lcd.setCursor(0, 162);
                        M5.Lcd.printf("jerk corr : %f", j_gyroyz_corr);
                        delay(2000); */

                        gyroXv.clear();
                        gyroYv.clear();
                        gyroZv.clear();
                        j_gyroXv.clear();
                        j_gyroYv.clear();
                        j_gyroZv.clear();
                        accXv.clear();
                        accYv.clear();                  
                        accZv.clear();
                        j_accXv.clear();
                        j_accYv.clear();                  
                        j_accZv.clear();
                        body_accXv.clear();
                        body_accYv.clear();
                        body_accZv.clear();
                        grav_accXv.clear();
                        grav_accYv.clear();
                        grav_accZv.clear();
                        mag_body_acc_v.clear();
                        mag_grav_acc_v.clear();
                        mag_body_j_acc_v.clear();
                        mag_gyro_v.clear();
                        mag_j_gyro_v.clear();

                        //features.push_back(mean_accx);
                        //features.push_back(mean_accy);
                        //features.push_back(mean_accz);
                        features.push_back(accx_std);
                        features.push_back(accy_std);
                        features.push_back(accz_std);
                        features.push_back(accx_mad);
                        features.push_back(accy_mad);
                        features.push_back(accz_mad);
                        //features.push_back(accx_max);
                        //features.push_back(accy_max);
                        //features.push_back(accz_max);
                        //features.push_back(accx_min);
                        //features.push_back(accy_min);
                        //features.push_back(accz_min);
                        features.push_back(acc_sma);
                        features.push_back(accx_energy);
                        features.push_back(accy_energy);
                        features.push_back(accz_energy);
                        features.push_back(accx_iqr);
                        features.push_back(accy_iqr);
                        features.push_back(accz_iqr);
                        //features.push_back(accx_entr);
                        //features.push_back(accy_entr);
                        //features.push_back(accz_entr);
                        for (size_t i = 0; i < accx_ar_c.size() - 1; i++)
                        {
                            features.push_back(accx_ar_c[i]);
                        }
                        for (size_t i = 0; i < accy_ar_c.size() - 1; i++)
                        {
                            features.push_back(accy_ar_c[i]);
                        }
                        for (size_t i = 0; i < accz_ar_c.size() - 1; i++)
                        {
                            features.push_back(accz_ar_c[i]);
                        }
                        //features.push_back(accxy_corr);
                        //features.push_back(accxz_corr);
                        //features.push_back(accyz_corr);
                        //pushing gravity
                        //features.push_back(mean_g_accx);
                        //features.push_back(mean_g_accy);
                        //features.push_back(mean_g_accz);
                        features.push_back(g_accx_std);
                        features.push_back(g_accy_std);
                        features.push_back(g_accz_std);
                        features.push_back(g_accx_mad);
                        features.push_back(g_accy_mad);
                        features.push_back(g_accz_mad);
                        features.push_back(g_accx_max);
                        //features.push_back(g_accy_max);
                        features.push_back(g_accz_max);
                        //features.push_back(g_accx_min);
                        //features.push_back(g_accy_min);
                        //features.push_back(g_accz_min);
                        features.push_back(g_acc_sma);
                        features.push_back(g_accx_energy);
                        features.push_back(g_accy_energy);
                        features.push_back(g_accz_energy);
                        features.push_back(g_accx_iqr);
                        features.push_back(g_accy_iqr);
                        features.push_back(g_accz_iqr);
                        //features.push_back(g_accx_entr);
                        //features.push_back(g_accy_entr);
                        //features.push_back(g_accz_entr);
                        for (size_t i = 0; i < g_accx_ar_c.size() - 1; i++)
                        {
                            features.push_back(g_accx_ar_c[i]);
                        }
                        for (size_t i = 0; i < g_accy_ar_c.size() - 1; i++)
                        {
                            features.push_back(g_accy_ar_c[i]);
                        }
                        for (size_t i = 0; i < g_accz_ar_c.size() - 1; i++)
                        {
                            features.push_back(g_accz_ar_c[i]);
                        }
                        //features.push_back(g_accxy_corr);
                        //features.push_back(g_accxz_corr);
                        //features.push_back(g_accyz_corr);
                        // pushing Acc Jerk
                        features.push_back(j_mean_accx);
                        //features.push_back(j_mean_accy);
                        features.push_back(j_mean_accz);
                        //features.push_back(j_accx_std);
                        //features.push_back(j_accy_std);
                        //features.push_back(j_accz_std);
                        features.push_back(j_accx_mad);
                        features.push_back(j_accy_mad);
                        features.push_back(j_accz_mad);
                        //features.push_back(j_accx_max);
                        //features.push_back(j_accy_max);
                        //features.push_back(j_accz_max);
                        //features.push_back(j_accx_min);
                        //features.push_back(j_accy_min);
                        //features.push_back(j_accz_min);
                        features.push_back(j_acc_sma);
                        features.push_back(j_accx_energy);
                        features.push_back(j_accy_energy);
                        features.push_back(j_accz_energy);
                        //features.push_back(j_accx_iqr);
                        //features.push_back(j_accy_iqr);
                        features.push_back(j_accz_iqr);
                        //features.push_back(j_accx_entr);
                        //features.push_back(j_accy_entr);
                        //features.push_back(j_accz_entr);
                        for (size_t i = 1; i < j_accx_ar_c.size(); i++)
                        {
                            features.push_back(j_accx_ar_c[i]);
                        }
                        for (size_t i = 1; i < j_accy_ar_c.size(); i++)
                        {
                            features.push_back(j_accy_ar_c[i]);
                        }
                        for (size_t i = 1; i < j_accz_ar_c.size(); i++)
                        {
                            features.push_back(j_accz_ar_c[i]);
                        }
                        //features.push_back(j_accxy_corr);
                        //features.push_back(j_accxz_corr);
                        //features.push_back(j_accyz_corr);
                        //pushing Gyro
                        //features.push_back(mean_gyrox);
                        //features.push_back(mean_gyroy);
                        //features.push_back(mean_gyroz);
                        //features.push_back(gyrox_std);
                        //features.push_back(gyroy_std);
                        //features.push_back(gyroz_std);
                        //features.push_back(gyrox_mad);
                        //features.push_back(gyroy_mad);
                        //features.push_back(gyroz_mad);
                        //features.push_back(gyrox_max);
                        //features.push_back(gyroy_max);
                        //features.push_back(gyroz_max);
                        //features.push_back(gyrox_min);
                        //features.push_back(gyroy_min);
                        //features.push_back(gyroz_min);
                        features.push_back(gyro_sma);
                        //features.push_back(gyrox_energy);
                        //features.push_back(gyroy_energy);
                        //features.push_back(gyroz_energy);
                        //features.push_back(gyrox_iqr);
                        //features.push_back(gyroy_iqr);
                        //features.push_back(gyroz_iqr);
                        //features.push_back(gyrox_entr);
                        //features.push_back(gyroy_entr);
                        //features.push_back(gyroz_entr);
                        for (size_t i = 0; i < gyrox_ar_c.size() - 1; i++)
                        {
                            features.push_back(gyrox_ar_c[i]);
                        }
                        for (size_t i = 0; i < gyroy_ar_c.size() - 1; i++)
                        {
                            features.push_back(gyroy_ar_c[i]);
                        }
                        for (size_t i = 0; i < gyroz_ar_c.size() - 1; i++)
                        {
                            features.push_back(gyroz_ar_c[i]);
                        }
                        //features.push_back(gyroxy_corr);
                        //features.push_back(gyroxz_corr);
                        //features.push_back(gyroyz_corr);
                        // pushing Gyro Jerk
                        //features.push_back(j_mean_gyrox);
                        //features.push_back(j_mean_gyroy);
                        //features.push_back(j_mean_gyroz);
                        //features.push_back(j_gyrox_std);
                        //features.push_back(j_gyroy_std);
                        //features.push_back(j_gyroz_std);
                        //features.push_back(j_gyrox_mad);
                        //features.push_back(j_gyroy_mad);
                        //features.push_back(j_gyroz_mad);
                        //features.push_back(j_gyrox_max);
                        //features.push_back(j_gyroy_max);
                        //features.push_back(j_gyroz_max);
                        //features.push_back(j_gyrox_min);
                        //features.push_back(j_gyroy_min);
                        //features.push_back(j_gyroz_min);
                        //features.push_back(j_gyro_sma);
                        //features.push_back(j_gyrox_energy);
                        //features.push_back(j_gyroy_energy);
                        //features.push_back(j_gyroz_energy);
                        //features.push_back(j_gyrox_iqr);
                        //features.push_back(j_gyroy_iqr);
                        //features.push_back(j_gyroz_iqr);
                        //features.push_back(j_gyrox_entr);
                        //features.push_back(j_gyroy_entr);
                        //features.push_back(j_gyroz_entr);
                        for (size_t i = 1; i < j_gyrox_ar_c.size(); i++)
                        {
                            features.push_back(j_gyrox_ar_c[i]);
                        }
                        for (size_t i = 1; i < j_gyroy_ar_c.size(); i++)
                        {
                            features.push_back(j_gyroy_ar_c[i]);
                        }
                        for (size_t i = 1; i < j_gyroz_ar_c.size(); i++)
                        {
                            features.push_back(j_gyroz_ar_c[i]);
                        }
                        features.push_back(j_gyroxy_corr);
                        features.push_back(j_gyroxz_corr);
                        features.push_back(j_gyroyz_corr);
                        // pushing Acc Mag
                        //features.push_back(mag_mean_body_acc);
                        features.push_back(mag_body_acc_std);
                        features.push_back(mag_body_acc_mad);
                        //features.push_back(mag_body_acc_max);
                        //features.push_back(mag_body_acc_min);
                        features.push_back(mag_body_acc_sma);
                        features.push_back(mag_body_acc_energy);
                        features.push_back(mag_body_acc_iqr);
                        //features.push_back(mag_body_acc_entr);
                        for (size_t i = 1; i < mag_body_acc_ar_c.size(); i++)
                        {
                            features.push_back(mag_body_acc_ar_c[i]);
                        }
                        // pushing Grav Mag
                        //features.push_back(mag_mean_grav_acc);
                        features.push_back(mag_grav_acc_std);
                        features.push_back(mag_grav_acc_mad);
                        //features.push_back(mag_grav_acc_max);
                        features.push_back(mag_grav_acc_min);
                        features.push_back(mag_grav_acc_sma);
                        features.push_back(mag_grav_acc_energy);
                        features.push_back(mag_grav_acc_iqr);
                        //features.push_back(mag_grav_acc_entr);
                        for (size_t i = 0; i < mag_grav_acc_ar_c.size() - 1; i++)
                        {
                            features.push_back(mag_grav_acc_ar_c[i]);
                        }
                        // pushing Acc Jerk Mag
                        features.push_back(mag_j_mean_body_acc);
                        features.push_back(mag_j_body_acc_std);
                        features.push_back(mag_j_body_acc_mad);
                        //features.push_back(mag_j_body_acc_max);
                        features.push_back(mag_j_body_acc_min);
                        features.push_back(mag_j_body_acc_sma);
                        features.push_back(mag_j_body_acc_energy);
                        //features.push_back(mag_j_body_acc_iqr);
                        //features.push_back(mag_j_body_acc_entr);
                        for (size_t i = 1; i < mag_j_body_acc_ar_c.size(); i = i + 2)
                        {
                            features.push_back(mag_j_body_acc_ar_c[i]);
                        }
                        // pushing Gyro Mag
                        features.push_back(mag_mean_gyro);
                        features.push_back(mag_gyro_std);
                        //features.push_back(mag_gyro_mad);
                        //features.push_back(mag_gyro_max);
                        //features.push_back(mag_gyro_min);
                        //features.push_back(mag_gyro_sma);
                        //features.push_back(mag_gyro_energy);
                        //features.push_back(mag_gyro_iqr);
                        //features.push_back(mag_gyro_entr);
                        for (size_t i = 0; i < mag_gyro_ar_c.size() - 3; i++)
                        {
                            features.push_back(mag_gyro_ar_c[i]);
                        }
                        // pushing Gyro Jerk Mag
                        //features.push_back(mag_j_mean_gyro);
                        features.push_back(mag_j_gyro_std);
                        features.push_back(mag_j_gyro_mad);
                        features.push_back(mag_j_gyro_max);
                        //features.push_back(mag_j_gyro_min);
                        //features.push_back(mag_j_gyro_sma);
                        //features.push_back(mag_j_gyro_energy);
                        //features.push_back(mag_j_gyro_iqr);
                        //features.push_back(mag_j_gyro_entr);
                        //for (size_t i = 0; i < mag_j_gyro_ar_c.size(); i++)
                        //{
                        //    features.push_back(mag_j_gyro_ar_c[i]);
                        //}
                    }

                    if (act_count == 8)
                    {
                        M5.Lcd.setCursor(0, 60);
                        M5.Lcd.printf("N_FEATURES: %d", features.size());
                        delay(5000);
                        int win_idx = winner_index(features);
                        int win_lbl = lbls[win_idx];
                        string win_act = menu_items[win_lbl];
                        M5.Lcd.setCursor(0, 202);
                        M5.Lcd.printf("winner index: %d", win_idx);
                        M5.Lcd.setCursor(0, 222);
                        M5.Lcd.printf("pred. act.: %s", win_act.c_str());
                        delay(5000);
                    }
                    
                    
                    if (act_count == 9)
                    {
                        String response;
                        if ((wifiMulti.run() == WL_CONNECTED)){
                            // Change the URL with the target server address
                            HTTPClient http;
                            string gx = vector_to_string(gyroXv);
                            gyroXv.clear();
                            string gy = vector_to_string(gyroYv);
                            gyroYv.clear();
                            string gz = vector_to_string(gyroZv);
                            gyroZv.clear();
                            string ax = vector_to_string(accXv);
                            accXv.clear();
                            string ay = vector_to_string(accYv);
                            accYv.clear();
                            string az = vector_to_string(accZv);
                            accZv.clear();
                            string url;
                            if (act_count == 9){
                                // check the URL of python server
                                url  = "http://scianso.pythonanywhere.com/som?gx=" + gx + "&gy=" + gy + "&gz=" + gz + "&ax=" + ax + "&ay=" + ay + "&az=" + az;
                            }
                            if (act_count == 7){
                                // check the URL of python server
                                url  = "http://scianso.pythonanywhere.com/kmeans?gx=" + gx + "&gy=" + gy + "&gz=" + gz + "&ax=" + ax + "&ay=" + ay + "&az=" + az;
                            }
                            http.begin(url.c_str());
                            //http.addHeader("Content-Type", "application/x-www-form-urlencoded");
                            M5.Lcd.clear();
                            int httpCode = http.GET();
                            if (httpCode > 0) {
                                response = http.getString();
                                //M5.Lcd.setCursor(0, 0);
                                //M5.Lcd.printf("response: %s", response.c_str());
                            } else {
                                M5.Lcd.clear();
                                M5.Lcd.setCursor(0, 162);
                                M5.Lcd.printf("[HTTP] GET... failed, error: %s\n", http.errorToString(httpCode).c_str());
                            }
                            http.end();
                            //delay(5000);
                            //M5.Lcd.clear();                  
                        }
                        features = split(response, ' ');
                        /* M5.Lcd.setCursor(0, 60);
                        M5.Lcd.printf("feature num: %d", features.size());
                        delay(2000);*/
                        /* for (size_t i = 0; i < N_WEIGHTS; i++)
                        {
                            M5.Lcd.setCursor(0, 60);
                            M5.Lcd.printf("label %d: %d", i, lbls[i]);
                            delay(2000);
                        } */
                        
                        int win_idx = winner_index(features);
                        int win_lbl = lbls[win_idx];
                        string win_act = menu_items[win_lbl];
                        M5.Lcd.setCursor(0, 202);
                        M5.Lcd.printf("winner index: %d", win_idx);
                        M5.Lcd.setCursor(0, 222);
                        M5.Lcd.printf("pred. act.: %s", win_act.c_str());
                        delay(5000);
                    }
                    
                    
                    
                    /* if ((wifiMulti.run() == WL_CONNECTED)){
                        // Change the URL with the target server address
                        HTTPClient http;
                        string str_feat = vector_to_string(features);
                        features.clear();
                        string url = "http://scianso.pythonanywhere.com/?data=" + str_feat;
                        if (act_count == 8){
                            url += "alg=som";
                        }
                        if (act_count == 7){
                            url += "alg=kmeans";
                        }
                        http.begin(url.c_str());
                        http.addHeader("Content-Type", "application/x-www-form-urlencoded");
                        int httpCode = http.GET();
                        if (httpCode > 0) {
                            String response = http.getString();
                            M5.Lcd.setCursor(0, 222);
                            M5.Lcd.printf("                 ");
                            M5.Lcd.setCursor(0, 222);
                            M5.Lcd.printf("response: %s", response.c_str());
                            delay(2000);
                        } else {
                            M5.Lcd.setCursor(0, 222);
                            M5.Lcd.printf("Error sending data");
                        }

                        http.end();
                    } */
                                        
                }
                else
                {
                    gyroXv.push_back(imu_data[0]);
                    gyroYv.push_back(imu_data[1]);
                    gyroZv.push_back(imu_data[2]);
                    accXv.push_back(imu_data[3]);
                    accYv.push_back(imu_data[4]);
                    accZv.push_back(imu_data[5]);
                    if (list_count == 1){
                        timev.push_back(0.00);
                    } else {
                        timev.push_back((1/41.29) * (list_count - 1));
                    }
                    delay(10);
                }
                /* if (M5.BtnA.wasPressed() || M5.BtnB.wasPressed() || M5.BtnC.wasPressed()){
                    execute = false;
                } */
            }
        }
    //delay(20);
    }
    if (clear_screen)
    {
        M5.Lcd.clear();
    }
}
