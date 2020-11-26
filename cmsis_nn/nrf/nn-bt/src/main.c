/*
 * Copyright (c) 2018 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-BSD-5-Clause-Nordic
 */

#include <zephyr/types.h>
#include <zephyr.h>
#include <drivers/uart.h>

#include <device.h>
#include <soc.h>

#include <bluetooth/bluetooth.h>
#include <bluetooth/uuid.h>
#include <bluetooth/gatt.h>
#include <bluetooth/hci.h>
#include <bluetooth/services/nus.h>
#include <settings/settings.h>
#include <stdio.h>
#include <logging/log.h>

#define LOG_MODULE_NAME peripheral_uart
LOG_MODULE_REGISTER(LOG_MODULE_NAME);

#define STACKSIZE 1024
#define PRIORITY 7

#define DEVICE_NAME CONFIG_BT_DEVICE_NAME
#define DEVICE_NAME_LEN	(sizeof(DEVICE_NAME) - 1)

#define UART_LABEL DT_LABEL(DT_NODELABEL(uart1))
#define UART_BUF_SIZE (10 * sizeof(uint8_t))

static K_SEM_DEFINE(ble_init_ok, 0, 1);
static struct bt_conn *current_conn;
static struct bt_conn *auth_conn;

static const struct device *uart;
static uint8_t uart_buf[UART_BUF_SIZE];
static struct k_poll_signal sig_ready = K_POLL_SIGNAL_INITIALIZER(sig_ready);

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};
static const struct bt_data sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_VAL),
};

static void uart_async_cb(const struct device *dev, struct uart_event *evt,
			  void *user_data)
{
	ARG_UNUSED(dev);

	if (evt->type == UART_RX_DISABLED) {
		printk("Got data over UART.\n");
		k_poll_signal_raise(&sig_ready, 0);
		uart_rx_enable(uart, uart_buf, sizeof(uart_buf),
			       SYS_FOREVER_MS);
	}
}

static void uart_init(void)
{
	uart = device_get_binding(UART_LABEL);
	__ASSERT(uart != NULL, "Unable to get UART device.");
	int err = uart_callback_set(uart, uart_async_cb, NULL);
	__ASSERT(err == 0, "Error %d setting UART callback", err);

}

static void connected(struct bt_conn *conn, uint8_t err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	if (err) {
		LOG_ERR("Connection failed (err %u)", err);
		return;
	}

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("Connected %s", log_strdup(addr));
	current_conn = bt_conn_ref(conn);
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("Disconnected: %s (reason %u)", log_strdup(addr), reason);

	if (auth_conn) {
		bt_conn_unref(auth_conn);
		auth_conn = NULL;
	}
	if (current_conn) {
		bt_conn_unref(current_conn);
		current_conn = NULL;
	}
}

static struct bt_conn_cb conn_callbacks = {
	.connected    = connected,
	.disconnected = disconnected,
};

void main(void)
{
	k_poll_signal_init(&sig_ready);
	uart_init();
	bt_conn_cb_register(&conn_callbacks);

	int err = bt_enable(NULL);
	__ASSERT(err == 0, "bt_enable returned %d.", err);

	LOG_INF("Bluetooth initialized");

	k_sem_give(&ble_init_ok);

	if (IS_ENABLED(CONFIG_SETTINGS)) {
		settings_load();
	}

	err = bt_nus_init(NULL);
	__ASSERT(err == 0, "Failed to initialize UART service (err: %d)", err);

	err = bt_le_adv_start(BT_LE_ADV_CONN, ad, ARRAY_SIZE(ad), sd,
			      ARRAY_SIZE(sd));
	__ASSERT(err == 0, "Advertising failed to start (err %d)", err);


	err = uart_rx_enable(uart, uart_buf, sizeof(uart_buf),
			     SYS_FOREVER_MS);
	__ASSERT(err == 0, "Unable to start UART RX (err %d)", err);

	printk("Starting Nordic UART service example\n");
}

void ble_write_thread(void)
{
	struct k_poll_event evt_ready = K_POLL_EVENT_INITIALIZER(
		K_POLL_TYPE_SIGNAL, K_POLL_MODE_NOTIFY_ONLY, &sig_ready);

	/* Don't go any further until BLE is initialized */
	k_sem_take(&ble_init_ok, K_FOREVER);

	for (;;) {
		/* Wait indefinitely for data to be sent over UART */
		k_poll(&evt_ready, 1, K_FOREVER);
		evt_ready.signal->signaled = 0;
		evt_ready.state = K_POLL_STATE_NOT_READY;
		printk("Forwarding data to BT.\n");
		if (bt_nus_send(NULL, uart_buf, sizeof(uart_buf))) {
			LOG_WRN("Failed to send data over BLE connection");
		}
	}
}

K_THREAD_DEFINE(ble_write_thread_id, STACKSIZE, ble_write_thread, NULL, NULL,
		NULL, PRIORITY, 0, 0);
